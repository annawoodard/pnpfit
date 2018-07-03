import atexit
import contextlib
import glob
import logging
import os
import sys
import tarfile
from collections import defaultdict
from datetime import datetime
import time

import jinja2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import scipy.ndimage
import seaborn as sns
import tabulate
from matplotlib.mlab import griddata
from matplotlib.ticker import FormatStrFormatter, LogLocator, MultipleLocator
from mpl_toolkits.axes_grid1 import ImageGrid
# from root_numpy import root2array
# import ROOT
from scipy.stats import chi2

from parsl.app.app import python_app
import pnpfit
from pnpfit.workspace import make_np_workspace
from pnpfit.util import cmssw_call

# from NPFit.NPFit import kde
# from NPFit.NPFit.makeflow import (fluctuate, max_likelihood_fit, multi_signal,
#                                   multidim_grid, multidim_np)
# from NPFit.NPFit.nll import fit_nll
# from NPFit.NPFit.parameters import conversion, label, nlo
# from NPFit.NPFit.scaling import load_fitted_scan
# from NPFitProduction.NPFitProduction.cross_sections import CrossSectionScan
# from NPFitProduction.NPFitProduction.utils import (cartesian_product,
#                                                    sorted_combos)

tweaks = {
    "lines.markeredgewidth": 0.0,
    "lines.linewidth": 3,
    "lines.markersize": 10,
    "patch.edgecolor": "black",
    "legend.facecolor": "white",
    "legend.frameon": True,
    "legend.edgecolor": "white",
    "legend.fontsize": "x-small",
    "legend.handletextpad": 0.5,
    "mathtext.fontset": "custom",
    # "mathtext.rm": "Bitstream Vera Sans",
    # "mathtext.it": "Bitstream Vera Sans:italic",
    # "mathtext.bf": "Bitstream Vera Sans:bold",
    "axes.labelsize": "small",
    "axes.titlesize": "small",
    "xtick.labelsize": "small",
    "xtick.major.size": 1,
    "ytick.major.size": 1,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "ytick.labelsize": "small",
}

x_min, x_max, y_min, y_max = np.array([0.200, 1.200, 0.550, 2.250])

def round(num, sig_figs):
    return str(float('{0:.{1}e}'.format(num, sig_figs - 1)))

def get_masked_colormap(bottom_map, top_map, norm, width, masked_value):
    low = masked_value - width / 2.
    high = masked_value + width / 2.
    if low > norm.vmin:
        colors = zip(np.linspace(0., norm(low), 100), bottom_map(np.linspace(0.1, 1., 100)))
        colors += [(norm(low), 'gray')]
    else:
        colors = [(0., 'gray')]
    if high < norm.vmax:
        colors += [(norm(high), 'gray')]
        colors += zip(np.linspace(norm(high), 1., 100), top_map(np.linspace(0.1, 1., 100)))
    else:
        colors += [(1., 'gray')]

    cmap = matplotlib.colors.LinearSegmentedColormap.from_list('masked_map', colors)

    return cmap

def get_stacked_colormaps(cmaps, interfaces, norm):
    colors = []
    low = 0.
    for cmap, interface in zip(cmaps, interfaces):
        colors += zip(np.linspace(low, norm(interface), 100), cmap(np.linspace(0, 1., 100)))
        low = norm(interface)

    colors += zip(np.linspace(low, 1., 100), cmaps[-1](np.linspace(0, 1., 100)))

    cmap = matplotlib.colors.LinearSegmentedColormap.from_list('masked_map', colors)

    return cmap

@contextlib.contextmanager
def saved_figure(x_label, y_label, name, outdir, header=False, figsize=(6.5, 6), dpi=None, luminosity=36):
    fig, ax = plt.subplots(figsize=figsize)
    lumi = str(luminosity) + ' fb$^{-1}$ (13 TeV)'
    # lumi = str(luminosity)
    if header:
        plt.title(lumi, loc='right', fontweight='normal', fontsize='x-small')
        plt.title(r'CMS', loc='left', fontweight='bold')
        if header == 'preliminary':
            plt.text(0.14, 1.02, r'Preliminary', style='italic', transform=ax.transAxes, fontsize='x-small')

    try:
        yield ax

    finally:
        logging.info('saving {}'.format(name))
        plt.xlabel(x_label, horizontalalignment='right', x=1.0)
        plt.ylabel(y_label, horizontalalignment='right', y=1.0)
        if dpi is None:
            plt.savefig(os.path.join(outdir, 'plots', '{}.pdf'.format(name)), bbox_inches='tight')
        else:
            plt.savefig(os.path.join(outdir, 'plots', '{}.pdf'.format(name)), bbox_inches='tight',
                    dpi=dpi)
        plt.savefig(os.path.join(outdir, 'plots', '{}.png'.format(name)), bbox_inches='tight',
                dpi=dpi)
        # plt.close()


class Plot(object):

    def __init__(self, subdir, outdir):
        self.subdir = subdir
        self.outdir = outdir

        if not os.path.isdir(os.path.join(outdir, 'plots', subdir)):
            os.makedirs(os.path.join(outdir, 'plots', subdir))

    def setup_inputs(self):
        """Make all inputs

        This method should add all of the commands producing input files which
        are needed by the Plot to the MakeflowSpecification. This command will
        be run for each plot in the analysis config file. By specifying this
        per-plot, commenting out plots in the config will remove their inputs
        from the Makefile, so that only the needed inputs are produced.

        """
        pass

    def write(self, config):
        """Write the plots

        This command should actually produce and save the plots.
        """
        try:
            os.makedirs(os.path.join(config['outdir'], 'plots', self.subdir))
        except OSError:
            pass  # the directory has already been made


def get_errs(scan, dimension, processes, config, clip=None):
    fits = None
    mgs = None
    for coefficients in sorted_combos(config['coefficients'], dimension):
        for process in processes:
            if process in scan.points[coefficients]:
                fit = scan.evaluate(coefficients, scan.points[coefficients][process], process)
                mg, _ = scan.scales(coefficients, process)

                if fits is None:
                    fits = fit
                    mgs = mg
                else:
                    fits = np.concatenate([fits, fit])
                    mgs = np.concatenate([mgs, mg])
    percent_errs = (mgs - fits) / mgs * 100
    np.random.shuffle(percent_errs)
    percent_errs = percent_errs[:clip]
    avg_abs_percent_errs = sum(np.abs(percent_errs)) / len(mgs[:clip])

    return percent_errs, avg_abs_percent_errs


class NewPhysicsScaling(Plot):

    def __init__(
            self,
            processes=[('ttZ', '+', '#2fd164')],
            subdir='scaling',
            overlay_result=False,
            dimensionless=False,
            match_nll_window=True,
            points=300):
        self.subdir = subdir
        self.processes = processes
        self.overlay_result = overlay_result
        self.dimensionless = dimensionless
        self.match_nll_window = match_nll_window
        self.points = points

    def setup_inputs(self, config, futures):
        inputs = ['cross_sections.npz']
        if self.match_nll_window:
            inputs = multidim_np(config, spec, 1, points=self.points)

        for coefficient in config['coefficients']:
            spec.add(inputs, [], ['run', 'plot', '--coefficients', coefficient, '--index', index, config['fn']])

    def write(self, config, plotter, args):
        super(NewPhysicsScaling, self).write(config)
        # FIXME does this work letting WQ transfer?
        scan = CrossSectionScan(os.path.join(config['outdir'], 'cross_sections.npz'))
        if self.match_nll_window:
            nll = fit_nll(config, transform=False, dimensionless=self.dimensionless)

        for coefficient in args.coefficients:
            conv = 1. if self.dimensionless else conversion[coefficient]
            if not np.any([p in scan.points[coefficient] for p, _, _ in self.processes]):
                continue
            with saved_figure(
                    label[coefficient] + ('' if self.dimensionless else r' $/\Lambda^2$'),
                    '$\sigma_{NP+SM} / \sigma_{SM}$',
                    os.path.join(self.subdir, coefficient)) as ax:

                for process, marker, c in self.processes:
                    x = scan.points[coefficient][process]
                    y, errs = scan.scales(coefficient, process)
                    if self.match_nll_window:
                        xmin = nll[coefficient]['x'][nll[coefficient]['y'] < 13].min()
                        xmax = nll[coefficient]['x'][nll[coefficient]['y'] < 13].max()
                    else:
                        xmin = min(x * conv)
                        xmax = max(x * conv)

                    xi = np.linspace(xmin, xmax, 10000).reshape(10000, 1)
                    yi = scan.evaluate(coefficient, xi, process)
                    ax.plot(xi * conv, yi, color='#C6C6C6')
                    ax.plot(x * conv, y, marker, mfc='none', markeredgewidth=2, markersize=15, label=label[process],
                            color=c)

                if self.overlay_result:
                    colors = ['black', 'gray']
                    for (x, _), color in zip(nll[coefficient]['best fit'], colors):
                        plt.axvline(
                            x=x,
                            ymax=0.5,
                            linestyle='-',
                            color=color,
                            label='Best fit\n {}$={:.2f}$'.format(label[coefficient], x)
                        )
                    for (low, high), color in zip(nll[coefficient]['one sigma'], colors):
                        plt.axvline(
                            x=low,
                            ymax=0.5,
                            linestyle='--',
                            color=color,
                            label='$1 \sigma [{:03.2f}, {:03.2f}]$'.format(low, high)
                        )
                        plt.axvline(
                            x=high,
                            ymax=0.5,
                            linestyle='--',
                            color=color
                        )
                    for (low, high), color in zip(nll[coefficient]['two sigma'], colors):
                        plt.axvline(
                            x=low,
                            ymax=0.5,
                            linestyle=':',
                            color=color,
                            label='$2 \sigma [{:03.2f}, {:03.2f}]$'.format(low, high)
                        )
                        plt.axvline(
                            x=high,
                            ymax=0.5,
                            linestyle=':',
                            color=color
                        )

                plt.xlim(xmin=xmin, xmax=xmax)
                plt.ylim(ymin=0, ymax=3)
                plt.title(r'CMS Simulation', loc='left', fontweight='bold')
                plt.title(r'MG5_aMC@NLO LO', loc='right', size=27)
                ax.legend(loc='upper center')
                if self.match_nll_window:
                    ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))


class NLL(Plot):
    def __init__(
            self,
            coefficient,
            cross_sections,
            processes,
            subdir='nll',
            transform=True,
            dimensionless=False,
            points=300,
            chunksize=300,
            asimov=False,
            outdir=os.path.abspath('results'),
            header='preliminary'):
        super().__init__(subdir, outdir)
        self.coefficient = coefficient
        self.cross_sections = cross_sections
        self.processes = processes
        self.transform = transform
        self.dimensionless = dimensionless
        self.points = points
        self.chunksize = chunksize
        self.asimov = asimov
        self.header = header


    def prepare_inputs(self):
        workspace = make_np_workspace(self.coefficient, self.outdir, self.cross_sections, self.processes)

        command = [
                'combine', '-M', 'MultiDimFit',
                workspace.filepath,
                '--setParameters {}=0.0'.format(self.coefficient),
                '--autoRange=20'
        ]
        if self.asimov:
            command += ['-t', '-1']

        scans = []
        for index in range(int(np.ceil(self.points / self.chunksize))):
            lowers = np.arange(1, self.points, self.chunksize)
            uppers = np.arange(self.chunksize, self.points + self.chunksize, self.chunksize)
            first, last = list(zip(lowers, uppers))[index]
            scan = os.path.join(self.outdir, 'scans', '{}_{}.root'.format(self.coefficient, index))
            postfix = [
                '--algo=grid',
                '--points={}'.format(self.points),
                '--firstPoint={}'.format(first),
                '--lastPoint={}'.format(last),
            ]
            scans.append(
                cmssw_call(
                    command + postfix,
                    inputs=[workspace],
                    outputs=[scan],
                    rename=['higgsCombineTest.MultiDimFit.mH120.root->{}'.format(scan)]
                ).outputs[0]
            )

        total = os.path.join(self.outdir, 'scans', '{}.total.root'.format(self.coefficient))
        command = ['hadd', '-f', total] + [f.filepath for f in scans]
        total = cmssw_call(
            command,
            inputs=scans,
            outputs=[os.path.join(self.outdir, 'scans', '{}.total.root'.format(self.coefficient))]
        ).outputs[0]

        command = [
            'python',
            os.path.join(pnpfit.__path__[0], 'nll.py'),
            '--coefficients'
        ] + [self.coefficient] + [
            '--outdir', self.outdir,
            '--xsecs', self.cross_sections,
            '--processes'
        ] + self.processes
        if self.transform:
            command += ['--transform']
        if self.dimensionless:
            command += ['--dimensionless']
        if self.asimov:
            command += ['--asimov']

        outpath = os.path.join(
            self.outdir, 'nll_{}_{}{}.npy'.format(
                self.coefficient,
                'transformed' if self.transform else '',
                '_dimensionless' if self.dimensionless else ''
            )
        )
        nll = cmssw_call(command, inputs=[total], outputs=[outpath]).outputs[0]

        self.inputs = [nll]

    def plot(self):
        @python_app(executors=['local'])
        def plot(data, cross_sections, coefficient, outdir, subdir, asimov, header, processes):
            from pnpfit.cross_sections import CrossSectionScan
            import numpy as np
            import matplotlib.pyplot as plt

            data = np.load(data.filepath, encoding='latin1')[()]
            scan = CrossSectionScan(cross_sections)

            info = data[coefficient]
            for p in processes:
                s0, s1, s2 = scan.construct(p, [coefficient])
                if not ((s1 > 1e-5) or (s2 > 1e-5)):
                    continue  # coefficient has no effect on any of the scaled processes
            x_label = '{} {}'.format(info['label'].replace('\ \mathrm{TeV}^{-2}', ''), info['units'])

            with saved_figure(
                    x_label,
                    '$-2\ \Delta\ \mathrm{ln}\ \mathrm{L}$' + (' (asimov data)' if asimov else ''),
                    os.path.join(subdir, coefficient),
                    outdir,
                    header=header) as ax:
                ax.plot(info['x'], info['y'], color='black')

                for i, (x, y) in enumerate(info['best fit']):
                    if i == 0:
                        plt.axvline(
                            x=x,
                            ymax=0.5,
                            linestyle=':',
                            color='black',
                            label='Best fit',
                        )
                    else:
                        plt.axvline(
                            x=x,
                            ymax=0.5,
                            linestyle=':',
                            color='black',
                        )
                for i, (low, high) in enumerate(info['one sigma']):
                    ax.plot(
                        [low, high],
                        [1.0, 1.0],
                        '--',
                        label=r'68% CL' if (i == 0) else '',
                        color='blue'
                    )
                for i, (low, high) in enumerate(info['two sigma']):
                    ax.plot(
                        [low, high],
                        [3.84, 3.84], 
                        '-.',
                        label=r'95% CL' if (i == 0) else '', color='#ff321a')

                ax.legend(loc='upper center')
                ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
                plt.ylim(ymin=0, ymax=12)
                plt.xlim(xmin=info['x'][info['y'] < 13].min(), xmax=info['x'][info['y'] < 13].max())
                if info['transformed']:
                    plt.xlim(xmin=0)

        self.future = plot(self.inputs[0], self.cross_sections, self.coefficient, self.outdir, self.subdir, self.asimov, self.header, self.processes)


# def prepare_nll_inputs(
#         coefficient,
#         cross_sections,
#         processes,
#         subdir='nll',
#         transform=True,
#         dimensionless=False,
#         points=100,
#         chunksize=100,
#         asimov=False,
#         outdir=os.path.abspath('results'),
#         header='preliminary'):
#     workspace = make_np_workspace(coefficient, outdir, cross_sections, processes)

#     command = [
#             'combine', '-M', 'MultiDimFit',
#             workspace.filepath,
#             '--setParameters {}=0.0'.format(coefficient),
#             '--autoRange=20'
#     ]
#     if asimov:
#         command += ['-t', '-1']

#     scans = []
#     for index in range(int(np.ceil(points / chunksize))):
#         lowers = np.arange(1, points, chunksize)
#         uppers = np.arange(chunksize, points + chunksize, chunksize)
#         first, last = list(zip(lowers, uppers))[index]
#         scan = os.path.join(outdir, 'scans', '{}_{}.root'.format(coefficient, index))
#         postfix = [
#             '--algo=grid',
#             '--points={}'.format(points),
#             '--firstPoint={}'.format(first),
#             '--lastPoint={}'.format(last),
#         ]
#         scans.append(
#             cmssw_call(
#                 command + postfix,
#                 inputs=[workspace],
#                 outputs=[scan],
#                 rename=['higgsCombineTest.MultiDimFit.mH120.root->{}'.format(scan)]
#             ).outputs[0]
#         )

#     total = os.path.join(outdir, 'scans', '{}.total.root'.format(coefficient))
#     command = ['hadd', '-f', total] + [f.filepath for f in scans]
#     total = cmssw_call(
#         command,
#         inputs=scans,
#         outputs=[os.path.join(outdir, 'scans', '{}.total.root'.format(coefficient))]
#     ).outputs[0]

#     command = [
#         'python',
#         os.path.join(pnpfit.__path__[0], 'nll.py'),
#         '--coefficients'
#     ] + [coefficient] + [
#         '--outdir', outdir,
#         '--xsecs', cross_sections,
#         '--processes'
#     ] + processes
#     if transform:
#         command += ['--transform']
#     if dimensionless:
#         command += ['--dimensionless']
#     if asimov:
#         command += ['--asimov']

#     outpath = os.path.join(
#         outdir, 'nll_{}_{}{}.npy'.format(
#             coefficient,
#             'transformed' if transform else '',
#             '_dimensionless' if dimensionless else ''
#         )
#     )
#     nll = cmssw_call(command, inputs=[total], outputs=[outpath]).outputs[0]

#     return [nll]


# def plot_nll(
#         inputs,
#         coefficient,
#         cross_sections,
#         processes,
#         subdir='nll',
#         transform=True,
#         dimensionless=False,
#         points=100,
#         chunksize=100,
#         asimov=False,
#         outdir=os.path.abspath('results'),
#         header='preliminary'):
#     # @python_app(executors=['local'])
#     @python_app
#     def plot(data, cross_sections, coefficient, outdir, subdir, asimov, header, processes):
#         from pnpfit.cross_sections import CrossSectionScan
#         import numpy as np
#         import matplotlib.pyplot as plt

#         data = np.load(data.filepath, encoding='latin1')[()]
#         scan = CrossSectionScan(cross_sections)

#         info = data[coefficient]
#         for p in processes:
#             s0, s1, s2 = scan.construct(p, [coefficient])
#             if not ((s1 > 1e-5) or (s2 > 1e-5)):
#                 continue  # coefficient has no effect on any of the scaled processes
#         x_label = '{} {}'.format(info['label'].replace('\ \mathrm{TeV}^{-2}', ''), info['units'])

#         with saved_figure(
#                 x_label,
#                 '$-2\ \Delta\ \mathrm{ln}\ \mathrm{L}$' + (' (asimov data)' if asimov else ''),
#                 os.path.join(subdir, coefficient),
#                 outdir,
#                 header=header) as ax:
#             ax.plot(info['x'], info['y'], color='black')

#             for i, (x, y) in enumerate(info['best fit']):
#                 if i == 0:
#                     plt.axvline(
#                         x=x,
#                         ymax=0.5,
#                         linestyle=':',
#                         color='black',
#                         label='Best fit',
#                     )
#                 else:
#                     plt.axvline(
#                         x=x,
#                         ymax=0.5,
#                         linestyle=':',
#                         color='black',
#                     )
#             for i, (low, high) in enumerate(info['one sigma']):
#                 ax.plot([low, high], [1.0, 1.0], '--', label=r'68% CL' if (i == 0) else '', color='blue')
#             for i, (low, high) in enumerate(info['two sigma']):
#                 ax.plot([low, high], [3.84, 3.84], '-.', label=r'95% CL' if (i == 0) else '', color='#ff321a')

#             ax.legend(loc='upper center')
#             ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
#             plt.ylim(ymin=0, ymax=12)
#             plt.xlim(xmin=info['x'][info['y'] < 13].min(), xmax=info['x'][info['y'] < 13].max())
#             if info['transformed']:
#                 plt.xlim(xmin=0)

#     return plot(inputs[0], cross_sections, coefficient, outdir, subdir, asimov, header, processes)


def two_signal_best_fit(config, ax, signals, theory_errors, tag, contours):
    # TODO switch this to mixin with to signal and contour method
    limits = root2array(os.path.join(config['outdir'], 'best-fit-{}.root'.format(tag)))

    x = limits['r_{}'.format(signals[0])] * nlo[signals[0]]
    y = limits['r_{}'.format(signals[1])] * nlo[signals[1]]
    z = 2 * limits['deltaNLL']

    if contours:
        levels = {
            2.30: '  1 $\sigma$',
            5.99: '  2 $\sigma$',
            # 11.83: ' 3 $\sigma$',
            # 19.33: ' 4 $\sigma$',
            # 28.74: ' 5 $\sigma$'
        }

        xi = np.linspace(x_min, x_max, 1000)
        yi = np.linspace(y_min, y_max, 1000)
        zi = griddata(x, y, z, xi, yi, interp='linear')

        cs = plt.contour(xi, yi, zi, sorted(levels.keys()), colors='black', linewidths=2)
        plt.clabel(cs, fmt=levels)

    handles = []
    labels = []

    bf, = plt.plot(
        x[z.argmin()],
        y[z.argmin()],
        color='black',
        mew=3,
        markersize=17,
        marker="*",
        linestyle='None'
    )
    handles.append(bf)
    labels.append('2D best fit')

    if theory_errors:
        x_process = signals[0]
        y_process = signals[1]
        xerr_low, xerr_high = np.array(theory_errors[y_process]) * nlo[y_process]
        yerr_low, yerr_high = np.array(theory_errors[x_process]) * nlo[x_process]
        theory = plt.errorbar(
            nlo[x_process], nlo[y_process],
            yerr=[[xerr_low], [xerr_high]],
            xerr=[[yerr_low], [yerr_high]],
            capsize=5,
            mew=2,
            color='black',
            ls='',
            marker='o',
            markersize=10,
            linewidth=3
        )
        handles.append(theory)
        labels.append('{} theory\n[1610.07922]'.format(label['ttV']))

    ax.set_xlim([x_min, x_max])
    ax.set_autoscalex_on(False)

    return handles, labels


# class TwoProcessCrossSectionSM(Plot):

#     def __init__(self, subdir='.', signals=['ttW', 'ttZ'], theory_errors=None, tag=None, numpoints=500, chunksize=500, contours=True):
#         self.subdir = subdir
#         self.signals = signals
#         self.theory_errors = theory_errors
#         if tag:
#             self.tag = tag
#         else:
#             self.tag = '-'.join(signals)
#         self.numpoints = numpoints
#         self.chunksize = chunksize
#         self.contours = contours

#     def setup_inputs(self, config, spec, index):
#         inputs = multi_signal(self.signals, self.tag, spec, config)
#         for signal in self.signals:
#             inputs += max_likelihood_fit(signal, spec, config)
#         if self.contours:
#             inputs += multidim_grid(config, self.tag, self.numpoints, self.chunksize, spec)

#         spec.add(inputs, [],  ['run', 'plot', '--index', index, config['fn']])

#     def write(self, config, plotter, args):
#         x = self.signals[0]
#         y = self.signals[1]
#         with plotter.saved_figure(
#                 label['sigma {}'.format(x)],
#                 label['sigma {}'.format(y)],
#                 self.tag,
#                 header=config['header']) as ax:
#             handles, labels = two_signal_best_fit(config, ax, self.signals, self.theory_errors, self.tag, self.contours)

#             data = root2array(os.path.join(config['outdir'], 'best-fit-{}.root'.format(x)))

#             x_cross_section = plt.axvline(x=data['limit'][0] * nlo[x], color='black')
#             x_error = ax.axvspan(
#                 data['limit'][1] * nlo[x],
#                 data['limit'][2] * nlo[x],
#                 alpha=0.5,
#                 color='#FA6900',
#                 linewidth=0.0
#             )
#             handles.append((x_cross_section, x_error))
#             labels.append('{} 1D $\pm$ $1\sigma$'.format(label[x]))

#             data = root2array(os.path.join(config['outdir'], 'best-fit-{}.root'.format(y)))

#             y_cross_section = plt.axhline(y=data['limit'][0] * nlo[y], color='black')
#             y_error = ax.axhspan(
#                 data['limit'][1] * nlo[y],
#                 data['limit'][2] * nlo[y],
#                 color='#69D2E7',
#                 alpha=0.5,
#                 linewidth=0.0
#             )
#             handles.append((y_cross_section, y_error))
#             labels.append('{} 1D $\pm$ $1\sigma$'.format(label[y]))

#             plt.legend(handles, labels)


# class TwoProcessCrossSectionSMAndNP(Plot):

#     def __init__(self,
#              subdir='.',
#              signals=['ttW', 'ttZ'],
#              theory_errors=None,
#              tag=None,
#              transform=True,
#              dimensionless=False,
#              points=300):
#         self.subdir = subdir
#         self.signals = signals
#         self.theory_errors = theory_errors
#         if tag:
#             self.tag = tag
#         else:
#             self.tag = '-'.join(signals)
#         self.transform = transform
#         self.dimensionless = dimensionless
#         self.points = points

#     def setup_inputs(self, config, spec, index):
#         inputs = multi_signal(self.signals, self.tag, spec, config)
#         inputs += multidim_np(config, spec, 1, points=self.points)
#         inputs += fluctuate(config, spec)

#         spec.add(inputs, [],  ['run', 'plot', '--index', index, config['fn']])

#     def write(self, config, plotter, args):
#         x_proc = self.signals[0]
#         y_proc = self.signals[1]
#         nll = fit_nll(config, self.transform, self.dimensionless)

#         table = []
#         scales = ['r_{}'.format(x) for x in config['processes']]
#         for coefficient in args.coefficients:
#             data = np.load(os.path.join(config['outdir'], 'fluctuations-{}.npy'.format(coefficient)))[()]
#             if np.isnan(data['x_sec_{}'.format(x_proc)]).any() or np.isnan(data['x_sec_{}'.format(y_proc)]).any():
#                 print('skipping coefficient {} with nan fluctuations'.format(coefficient))
#                 continue

#             with plotter.saved_figure(
#                     label['sigma {}'.format(x_proc)],
#                     label['sigma {}'.format(y_proc)],
#                     os.path.join(self.subdir, '{}_{}'.format(self.tag, coefficient)),
#                     header=config['header']) as ax:
#                 handles, labels = two_signal_best_fit(config, ax, self.signals, self.theory_errors, self.tag, False)

#                 x = data['x_sec_{}'.format(x_proc)]
#                 y = data['x_sec_{}'.format(y_proc)]

#                 try:
#                     kdehist = kde.kdehist2(x, y, [70, 70])
#                     clevels = sorted(kde.confmap(kdehist[0], [.6827, .9545]))
#                     contour = ax.contour(kdehist[1], kdehist[2], kdehist[0], clevels, colors=['#ff321a', 'blue'], linestyles=['-.', '--'])
#                     for handle, l in zip(contour.collections[::-1], ['68% CL', '95% CL']):
#                         handles.append(handle)
#                         labels.append(l)
#                 except Exception as e:
#                     print('problem making contour for {}: {}'.format(coefficient, e))

#                 colors = ['black', 'gray']
#                 for (bf, _), color in zip(nll[coefficient]['best fit'], colors):
#                     table.append([coefficient, '{:.2f}'.format(bf), '{:.2f}'.format(data[0][coefficient])] + ['{:.2f}'.format(data[0][i]) for i in scales])
#                     point, = plt.plot(
#                         data[0]['x_sec_{}'.format(x_proc)],
#                         data[0]['x_sec_{}'.format(y_proc)],
#                         color=color,
#                         markeredgecolor=color,
#                         mew=3,
#                         markersize=17,
#                         marker="x",
#                         linestyle='None'
#                     )
#                     handles.append(point)

#                     labels.append("Best fit\n{}".format(nll[coefficient]['label']))
#                 plt.legend(handles, labels, loc='upper right', fontsize=27)
#                 plt.ylim(ymin=y_min, ymax=y_max)
#                 plt.xlim(xmin=x_min, xmax=x_max)
#                 ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
#                 ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))

#         print(tabulate.tabulate(table, headers=['coefficient', 'bf', 'coefficient value'] + config['processes']))

