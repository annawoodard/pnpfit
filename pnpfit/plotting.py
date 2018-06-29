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
matplotlib.use('Agg')
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
from root_numpy import root2array
import ROOT
from scipy.stats import chi2

from NPFit.NPFit import kde
from NPFit.NPFit.makeflow import (fluctuate, max_likelihood_fit, multi_signal,
                                  multidim_grid, multidim_np)
from NPFit.NPFit.nll import fit_nll
from NPFit.NPFit.parameters import conversion, label, nlo
from NPFit.NPFit.scaling import load_fitted_scan
from NPFitProduction.NPFitProduction.cross_sections import CrossSectionScan
from NPFitProduction.NPFitProduction.utils import (cartesian_product,
                                                   sorted_combos)

tweaks = {
    "lines.markeredgewidth": 0.0,
    "lines.linewidth": 8,
    "lines.markersize": 23,
    "patch.edgecolor": "black",
    "legend.facecolor": "white",
    "legend.frameon": True,
    "legend.edgecolor": "white",
    "legend.fontsize": "medium",
    "legend.handletextpad": 0.5,
    "mathtext.fontset": "custom",
    "mathtext.rm": "Bitstream Vera Sans",
    "mathtext.it": "Bitstream Vera Sans:italic",
    "mathtext.bf": "Bitstream Vera Sans:bold",
    "axes.labelsize": "x-large",
    "axes.titlesize": "x-large",
    "xtick.labelsize": "x-large",
    "xtick.major.size": 5,
    "ytick.major.size": 5,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "ytick.labelsize": "x-large",
}
sns.set(context="poster", style="white", font_scale=1.5, rc=tweaks)

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

class Plotter(object):

    def __init__(self, config):
        atexit.register(self.webify)
        self.config = config

    def webify(self):
        path = os.path.join(os.environ['LOCALRT'], 'src', 'NPFit', 'NPFit', 'data')
        env = jinja2.Environment(loader=jinja2.FileSystemLoader(path))
        env.filters["datetime"] = lambda d: datetime.fromtimestamp(d).strftime('%a, %d %b %Y, %H:%M')
        env.filters["basename"] = lambda d: os.path.basename(d)
        env.tests["sum"] = lambda s: s == "Total"
        template = env.get_template('template.html')

        tfilename = os.path.join(self.config['outdir'], 'plots', 'plots.tar.gz')
        tfile = tarfile.open(tfilename, 'w:gz')
        for root, dirs, files in os.walk(os.path.abspath(self.config['outdir'])):
            for path in [os.path.join(root, d) for d in dirs] + [root]:
                pngs = glob.glob(os.path.join(path, '*.png'))
                pdfs = glob.glob(os.path.join(path, '*.pdf'))
                plots = [os.path.splitext(os.path.basename(name))[0] for name in pngs]
                if plots:
                    plots.sort()
                    [tfile.add(f, arcname=os.path.basename(name)) for f in [os.path.join(path, name) for name in pdfs]]

                dirs = ['..']
                files = []
                names = [x for x in os.listdir(path) if not x.startswith('.') and '#' not in x and x != 'index.html']
                for name in names:
                    if '.png' not in name and '.pdf' not in name:
                        fullname = os.path.join(path, name)
                        if os.path.isdir(fullname):
                            if name not in ['javascripts', 'stylesheets']:
                                dirs.append(name)
                        else:
                            files.append({'name': name, 'timestamp': os.path.getmtime(fullname)})

                with open(os.path.join(path, 'index.html'), 'w') as f:
                    logging.info('updating index {}'.format(os.path.join(path, 'index.html')))
                    f.write(template.render(
                        dirs=sorted(dirs),
                        files=sorted(files),
                        plots=plots,
                    ).encode('utf-8'))

        tfile.close()

    @contextlib.contextmanager
    def saved_figure(self, x_label, y_label, name, header=False, figsize=(11, 11), dpi=None):
        fig, ax = plt.subplots(figsize=figsize)
        lumi = str(self.config['luminosity']) + ' fb$^{-1}$ (13 TeV)'
        if header:
            plt.title(lumi, loc='right', fontweight='normal')
            plt.title(r'CMS', loc='left', fontweight='bold')
            if header == 'preliminary':
                plt.text(0.155, 1.009, r'Preliminary', style='italic', transform=ax.transAxes)

        try:
            yield ax

        finally:
            logging.info('saving {}'.format(name))
            plt.xlabel(x_label, horizontalalignment='right', x=1.0)
            plt.ylabel(y_label, horizontalalignment='right', y=1.0)
            if dpi is None:
                plt.savefig(os.path.join(self.config['outdir'], 'plots', '{}.pdf'.format(name)), bbox_inches='tight')
            else:
                plt.savefig(os.path.join(self.config['outdir'], 'plots', '{}.pdf'.format(name)), bbox_inches='tight',
                        dpi=dpi)
            plt.savefig(os.path.join(self.config['outdir'], 'plots', '{}.png'.format(name)), bbox_inches='tight',
                    dpi=dpi)
            plt.close()


class Plot(object):

    def __init__(self, subdir):
        self.subdir = subdir

    def specify(self):
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


class FitQualityByPoints(Plot):

    def __init__(self, dimensions=[1], processes=['ttZ', 'ttH', 'ttW'], points=None, subdir='fit'):
        self.dimensions = dimensions
        self.processes = processes
        self.subdir = subdir
        if points is None:
            self.fitpoints = np.array(range(10, 1000, 100))
        else:
            self.fitpoints = np.array(points)

    def specify(self, config, spec, index):
        spec.add(['cross_sections.npz'], [], ['run', 'plot', '--index', str(index), config['fn']])

    def write(self, config, plotter, args):
        super(FitQualityByPoints, self).write(config)

        name = os.path.join(self.subdir, 'fit_quality_by_points')
        y_label = r'$\frac{100}{n}\sum^n_{i=1} |\frac{\mu_{\mathrm{MG}} - \mu_{\mathrm{fit}}}{\mu_{\mathrm{MG}}}|_i$'
        loc = MultipleLocator(5)
        with plotter.saved_figure('fit points', y_label, name,
                figsize=(17, 11)) as ax:
            labels = []
            table = []
            scan = load_fitted_scan(config, 'cross_sections.npz')
            testpoints = dict((dimension, 0) for dimension, _, _ in self.dimensions)
            for dimension, _, _ in self.dimensions:
                for k in scan.points.keys():
                    if len(k) == dimension:
                        testpoints[dimension] += sum([len(v) for v in scan.points[k].values()])
            avg_abs_percent_errs = np.zeros(len(self.fitpoints))

            for dimension, marker, color in self.dimensions:
                for index, points in enumerate(self.fitpoints):
                    scan = load_fitted_scan(config, 'cross_sections.npz', maxpoints=points, dimension=dimension)
                    errs, avg_abs_percent_err = get_errs(scan, dimension, self.processes, config,
                            clip=min(testpoints.values()))

                    table.append([points, avg_abs_percent_err, len(errs), dimension])
                    avg_abs_percent_errs[index] = avg_abs_percent_err

                ax.plot(self.fitpoints, avg_abs_percent_errs, marker=marker, markersize=10, linewidth=1, linestyle='none',
                        label='fit dimension {}'.format(dimension), color=color)
            ax.set_yscale('log', subsy=range(10))
            ax.xaxis.set_minor_locator(loc)
            ax.tick_params(axis='x', length=5, which='minor')
            plt.legend(fontsize='large')

        headers = ['fit points', 'average absolute percent error', 'total test points', 'dimension']
        with open(os.path.join(config['outdir'], 'fit_quality_by_points.txt'), 'w') as f:
            f.write(tabulate.tabulate(table, headers=headers) + '\n')

class FitQualityByDim(Plot):

    def __init__(self, fit_dimensions=[1], eval_dimensions=None, processes=['ttZ', 'ttH', 'ttW'], fit_to_test_ratio=1,
            subdir='fit', xmin=-100, xmax=50):
        self.fit_dimensions = fit_dimensions
        self.eval_dimensions = eval_dimensions
        self.processes = processes
        self.fit_to_test_ratio = float(fit_to_test_ratio)
        self.subdir = subdir
        self.xmin = xmin
        self.xmax = xmax

    def specify(self, config, spec, index):
        spec.add(['cross_sections.npz'], [], ['run', 'plot', '--index', str(index), config['fn']])

    def write(self, config, plotter, args):
        super(FitQualityByDim, self).write(config)

        name = os.path.join(self.subdir, 'fit_quality_by_dim')
        x_label = r'$100 \frac{\mu_{\mathrm{MG}} - \mu_{\mathrm{fit}}}{\mu_{\mathrm{MG}}}$'
        scan = CrossSectionScan(os.path.join(config['outdir'], 'cross_sections.npz'))

        testpoints = dict(('_'.join([str(d) for d in dimensions]), 0) for dimensions in self.fit_dimensions)
        for dimensions in self.fit_dimensions:
            tag = '_'.join([str(d) for d in dimensions])
            for k in scan.points.keys():
                if len(k) in dimensions:
                    testpoints[tag] += sum([len(v) for v in scan.points[k].values()])
        with plotter.saved_figure(x_label, 'counts', name) as ax:
            labels = []
            table = []

            for fit_dims in self.fit_dimensions:
                scan.fit(dimensions=fit_dims)
                for eval_dim in (self.eval_dimensions if self.eval_dimensions is not None else fit_dims):
                    errs, _ = get_errs(scan, eval_dim, self.processes, config, clip=min(testpoints.values()))
                    errs[errs < self.xmin] = self.xmin
                    errs[errs > self.xmax] = self.xmax
                    if self.eval_dimensions is None:
                        label = '{fdim} fit $\sigma$={s:0.2f}'.format(
                            fdim=' and '.join(('{}d'.format(dim) for dim in fit_dims)),
                            s=np.std(errs),
                        )
                    else:
                        label = '{fdim} fit, {edim}d evaluation $\sigma_{{{sfdim},{edim}}}$={s:0.2f}'.format(
                            fdim=' and '.join(('{}d'.format(dim) for dim in fit_dims)),
                            edim=eval_dim,
                            sfdim='+'.join((str(dim) for dim in fit_dims)),
                            s=np.std(errs),
                        )
                    ax.hist(errs, 70, histtype='step', fill=False, label=label, linewidth=4)
                    bad = errs[(errs < -5) | (errs > 5)]
                ax.set_yscale('log', subsy=range(10))
                if self.eval_dimensions is not None:
                    plt.ylim(ymin=0, ymax=2e7)
                    plt.xlim(xmin=-100)
                ax.yaxis.set_tick_params('minor', size=5)
                plt.legend(title='{:,} test points'.format(len(errs)), loc='upper left')


class NewPhysicsScaling2D(Plot):

    def __init__(
            self,
            processes=['ttZ', 'ttH', 'ttW'],
            subdir='scaling2d',
            dimensionless=False,
            dimensions=None,
            maxnll=None,
            match_zwindows=False,
            madgraph=False,
            numvalues=100,
            points=10000,
            dpi=500,
            profile=True):
        self.subdir = subdir
        self.processes = processes
        self.dimensionless = dimensionless
        self.dimensions = dimensions if dimensions is not None else [2]
        self.maxnll = maxnll
        self.match_zwindows = match_zwindows
        self.madgraph = madgraph
        self.numvalues = numvalues
        self.points = points
        self.dpi = dpi
        self.profile = profile

    def specify(self, config, spec, index):
        inputs = []
        if self.maxnll is not None:
            inputs += multidim_np(config, spec, 2, points=self.points)

        for coefficients in sorted_combos(config['coefficients'], 2):
            cmd = 'run plot --coefficients {coefficients} --index {index} {fn}'
            base = os.path.join(config['outdir'], 'plots', self.subdir, '_'.join(coefficients))
            outputs = [base + ext for ext in ['.pdf', '.png']]
            spec.add(inputs + ['cross_sections.npz'], outputs, cmd.format(coefficients=' '.join(coefficients), index=index, fn=config['fn']))
            if self.profile:
                workspace = os.path.join(config['outdir'], 'workspaces', '{}.root'.format('_'.join(config['coefficients'])))
                cmd = ['run', 'combine', '--snapshot'] + list(coefficients) + [config['fn']]
                snapshot = os.path.join(config['outdir'], 'snapshots', '{}.root'.format('_'.join(coefficients)))
                spec.add([workspace], [snapshot], cmd)

    def write(self, config, plotter, args):
        super(NewPhysicsScaling2D, self).write(config)
        scan = CrossSectionScan(os.path.join(config['outdir'], 'cross_sections.npz'))
        scan.fit(dimensions=self.dimensions)

        if self.match_zwindows:
            zmin = None
            zmax = None
            for coefficients in sorted_combos(args.coefficients, 2):
                madgraph = scan.dataframe(coefficients)
                if zmin is None:
                    zmin = min(madgraph[self.processes].min())
                    zmax = max(madgraph[self.processes].max())
                else:
                    zmin = min(min(madgraph[self.processes].min()), zmin)
                    zmax = max(max(madgraph[self.processes].max()), zmax)

        for coefficients in sorted_combos(args.coefficients, 2):
            tag = '_'.join(coefficients)
            name = os.path.join(self.subdir, tag)
            x = coefficients[0]
            y = coefficients[1]

            x_label = label[x] + ('' if self.dimensionless else r' $/\Lambda^2\, / \,\mathrm{TeV}^{-2}$')
            y_label = label[y] + ('' if self.dimensionless else r' $/\Lambda^2\, / \,\mathrm{TeV}^{-2}$')
            x_conv = 1. if self.dimensionless else conversion[x]
            y_conv = 1. if self.dimensionless else conversion[y]

            if coefficients in scan.points.keys():
                madgraph = scan.dataframe(coefficients)
            else:
                madgraph = scan.dataframe(sorted(config['coefficients']))
            xmin = madgraph[x].min()
            xmax = madgraph[x].max()
            ymin = madgraph[y].min()
            ymax = madgraph[y].max()

            if self.maxnll:
                try:
                    data = root2array(os.path.join(config['outdir'], 'scans', '{}.total.root'.format(tag)))
                except IOError as e:
                    print 'input data missing, will not match nll for {}'.format(tag)
                    continue

                zi = 2 * data['deltaNLL']
                xi = data[x]
                yi = data[y]
                xmin = xi[zi < self.maxnll].min()
                ymin = yi[zi < self.maxnll].min()
                xmax = xi[zi < self.maxnll].max()
                ymax = yi[zi < self.maxnll].max()
                window = (madgraph[x] > xmin) & (madgraph[x] < xmax) & (madgraph[y] > ymin) & (madgraph[y] < ymax)
                madgraph = madgraph[window]


            if not self.match_zwindows:
                zmin = min(madgraph[self.processes].min())
                zmax = max(madgraph[self.processes].max())

            norm = matplotlib.colors.LogNorm(vmin=min(zmin, 0.9), vmax=zmax)
            masked_map = get_masked_colormap(
                    sns.light_palette("navy", as_cmap=True),
                    sns.light_palette((210, 90, 60), input="husl", as_cmap=True),
                    norm,
                    0.1,
                    1.0
            )

            fig = plt.figure(figsize=(30, 9))
            grid = ImageGrid(
                fig, 111,
                nrows_ncols=(1, len(self.processes)),
                axes_pad=0.45,
                share_all=True,
                cbar_location="right",
                cbar_mode="single",
                cbar_size="7%",
                cbar_pad=0.15,
                aspect=False
            )

            offplane_coefficients = []
            offplane_values = []
            if self.madgraph:
                df = madgraph
                for c in sorted(config['coefficients']):
                    if c not in coefficients:
                        offplane_coefficients += [c]
                        offplane_values += [0.0]
            else:
                if self.profile:
                    f = ROOT.TFile(os.path.join(config['outdir'], 'snapshots',
                        '{}.root'.format('_'.join(coefficients))))
                    w = f.Get('w')
                    w.loadSnapshot('MultiDimFit')
                values = []
                columns = sorted(config['coefficients'])
                for column in columns:
                    if column in coefficients:
                        if column == x:
                            values += [np.linspace(xmin, xmax, self.numvalues)]
                        if column == y:
                            values += [np.linspace(ymin, ymax, self.numvalues)]
                    else:
                        offplane_coefficients += [column]
                        if self.profile:
                            var = w.var(column)
                            values += [np.array([var.getVal()])]
                            offplane_values += [var.getVal() * (1. if self.dimensionless else conversion[column])]
                        else:
                            values += [np.zeros(1)]
                            offplane_values += [0.]
                start = time.time()
                df = scan.dataframe(columns, evaluate_points=cartesian_product(*values))
                print('took {:.1f} seconds to get df '.format(time.time() - start))

            for ax, process in zip(grid, self.processes):
                columns = list(coefficients) + [process]
                data = df[columns]
                xx = data[x].values * x_conv
                yy = data[y].values * y_conv
                X, Y = np.meshgrid(np.linspace(xmin * x_conv, xmax * x_conv, self.numvalues * 10), np.linspace(ymin *
                    y_conv, ymax * y_conv, self.numvalues * 10))
                Z = scipy.interpolate.griddata((xx, yy), data[process].values, (X, Y), method='nearest')
                msize = 200 if self.madgraph else 25
                marker = 'o' if self.madgraph else 's'
                start = time.time()
                if self.madgraph:
                    scatter = ax.scatter(
                            xx,
                            yy,
                            c=data[process],
                            norm=norm,
                            s=msize,
                            marker=marker,
                            cmap=masked_map,
                            edgecolors='face',
                            rasterized=True
                    )
                else:
                    scatter = ax.pcolormesh(
                            X,
                            Y,
                            Z,
                            rasterized=True,
                            norm=norm,
                            cmap=masked_map)
                if not self.profile:
                    ax.scatter(
                            [0.0],
                            [0.0],
                            c='red',
                            s=600,
                            linewidths=0,
                            marker='*',
                            label='SM'
                    )
                ax.set_ylabel(y_label, horizontalalignment='right', y=1.0)
                ax.set_xlim([xx.min(), xx.max()])
                ax.set_ylim([yy.min(), yy.max()])
                ax.annotate(
                    label[process],
                    xy=(0.5, 0.85),
                    xycoords='axes fraction',
                    horizontalalignment='center',
                    fontsize='x-large',
                    bbox=dict(boxstyle="round,pad=.5", fc="white", ec="none")
                )

            bar = fig.colorbar(
                    scatter,
                    cax=ax.cax,
                    label='$\sigma_{NP+SM} / \sigma_{SM}$',
                    ticks=LogLocator(subs=range(10)),
                )
            # there is bug in this version of matplotlib ignores zorder, so redraw ticklines
            for t in ax.cax.yaxis.get_ticklines():
                ax.cax.add_artist(t)

            ax.legend(fancybox=True)
            if self.dimensionless:
                point = ', '.join([label[c] for c in offplane_coefficients])
            else:
                point = ', '.join(['$\\frac{{{}}}{{\Lambda^2}}$'.format(label[c].replace('$', '')) for c in offplane_coefficients])
            title = fig.suptitle(
                    '({}) = ({}){}'.format(
                        point,
                        ', '.join([round(v, 2) for v in offplane_values]),
                        ' TeV$^{-2}$' if not self.dimensionless else ''),
                    fontsize="x-large"
            )
            title.set_position([.5, 1.05])

            logging.info('saving {}'.format(name))
            ax.set_xlabel(x_label, horizontalalignment='right', x=1.0)
            start = time.time()
            plt.savefig(os.path.join(config['outdir'], 'plots', '{}.pdf'.format(name)), bbox_inches='tight', dpi=self.dpi)
            plt.savefig(os.path.join(config['outdir'], 'plots', '{}.png'.format(name)), bbox_inches='tight',
                    dpi=self.dpi)
            plt.close()
            print('took {:.1f} seconds to save plot '.format(time.time() - start))


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

    def specify(self, config, spec, index):
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
            with plotter.saved_figure(
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


class NLL2D(Plot):

    def __init__(self, subdir='nll2d', dimensionless=False, draw='mesh', maxnll=12, vmin=0.05, points=2000, dpi=400,
            freeze=False):
        self.subdir = subdir
        self.dimensionless = dimensionless
        if draw not in ['mesh', 'scatter', None]:
            raise NotImplementedError('can only draw mesh, scatter, or None')
        self.draw = draw
        self.maxnll = maxnll
        self.vmin = vmin
        self.points = points
        self.dpi = dpi
        self.freeze = freeze

    def specify(self, config, spec, index):
        multidim_np(config, spec, 2, points=self.points, freeze=self.freeze)

        for coefficients in sorted_combos(config['coefficients'], 2):
            label = '{}{}'.format('_'.join(coefficients), '_frozen' if self.freeze else '')
            inputs = [
                config['fn'],
                os.path.join(config['outdir'], 'scans', '{}.total.root'.format(label))
            ]
            cmd = 'run plot --coefficients {coefficients} --index {index} {fn}'
            outputs = [os.path.join(config['outdir'], 'plots', self.subdir, '{}.pdf'.format(label))]
            spec.add(inputs, outputs, cmd.format(coefficients=' '.join(coefficients), index=index, fn=config['fn']))

    def write(self, config, plotter, args):
        super(NLL2D, self).write(config)

        levels = sorted(chi2.isf([0.05, 0.32], 2))
        labels = ['68% CL', '95% CL']
        for coefficients in sorted_combos(args.coefficients, 2):
            tag = '{}{}'.format('_'.join(coefficients), '_frozen' if self.freeze else '')
            try:
                data = root2array(os.path.join(config['outdir'], 'scans', '{}.total.root'.format(tag)))
            except IOError as e:
                print 'input data missing, skipping {}'.format(tag)
                continue

            x = coefficients[0]
            y = coefficients[1]
            zi = 2 * data['deltaNLL']
            zi = zi - zi.min()
            xi = data[x]
            yi = data[y]
            x_label = label[x] + ('' if self.dimensionless else r' $/\Lambda^2$')
            y_label = label[y] + ('' if self.dimensionless else r' $/\Lambda^2$')
            if not self.dimensionless:
                xi *= conversion[x]
                yi *= conversion[y]
                x_label += '$\ [\mathrm{TeV}^{-2}]$'
                y_label += '$\ [\mathrm{TeV}^{-2}]$'

            xmin = xi[zi < self.maxnll].min()
            xmax = xi[zi < self.maxnll].max()
            ymin = yi[zi < self.maxnll].min()
            ymax = yi[zi < self.maxnll].max()
            window = (xi > xmin) & (xi < xmax) & (yi > ymin) & (yi < ymax)

            gridsize = 50
            X, Y = np.meshgrid(
                    np.linspace(xmin, xmax, gridsize),
                    np.linspace(ymin, ymax, gridsize)
            )
            zmin = 0.1
            xi = xi[window]
            yi = yi[window]
            zi = zi[window]
            bf_x = xi[zi.argmin()]
            bf_y = yi[zi.argmin()]

            Z = scipy.interpolate.griddata((xi, yi), zi, (X, Y), method='nearest')
            Z = scipy.ndimage.filters.gaussian_filter(Z, 1.6)
            with plotter.saved_figure(
                    x_label,
                    y_label,
                    os.path.join(self.subdir, tag),
                    header=config['header'],
                    figsize=(15, 11),
                    dpi=self.dpi) as ax:

                draw = self.draw if (x, y) != ('c2G', 'c3G') else 'scatter'
                if draw == 'mesh':
                    contour = plt.tricontour(
                            X.ravel(),
                            Y.ravel(),
                            Z.ravel(),
                            levels,
                            colors=['black', 'black'],
                            linestyles=['--', '-']
                    )
                else:
                    contour = plt.tricontour(
                            xi,
                            yi,
                            zi,
                            levels,
                            colors=['black', 'black'],
                            linestyles=['--', '-']
                    )

                for i, l in enumerate(labels):
                    contour.collections[i].set_label(l)

                plt.plot(
                    bf_x,
                    bf_y,
                    mew=3,
                    marker="x",
                    linestyle='None',
                    color='black',
                    label='best fit',
                    zorder=10
                )

                if self.draw in ['scatter', 'mesh']:
                    np.clip(Z, self.vmin, zi.max(), Z)
                    np.clip(zi, self.vmin, zi.max(), zi)

                    if draw is 'mesh':
                        points = ax.pcolormesh(
                                X,
                                Y,
                                Z,
                            norm=matplotlib.colors.LogNorm(),
                            cmap=get_stacked_colormaps(
                                [sns.light_palette("red", reverse=True, as_cmap=True), sns.light_palette("blue", as_cmap=True)],
                                interfaces=levels[:-1],
                                norm=matplotlib.colors.LogNorm(vmin=self.vmin, vmax=Z.max())
                            ),
                            rasterized=True
                        )
                    elif draw is 'scatter':
                        points = ax.scatter(
                            xi,
                            yi,
                            c=zi,
                            norm=matplotlib.colors.LogNorm(),
                            s=600,
                            marker='s',
                            linewidths=0,
                            cmap=get_stacked_colormaps(
                                [sns.light_palette("red", reverse=True, as_cmap=True), sns.light_palette("blue", as_cmap=True)],
                                interfaces=levels[:-1],
                                norm=matplotlib.colors.LogNorm(vmin=self.vmin, vmax=zi.max())
                            ),
                            rasterized=True
                        )
                    bar = plt.colorbar(
                        points,
                        label='$-2\ \Delta\ \mathrm{ln}\ \mathrm{L}$' + (' (asimov data)' if config['asimov data'] else ''),
                        ticks=LogLocator(subs=range(10))
                    )
                    for t in bar.ax.get_yticklines():
                        bar.ax.add_artist(t)
                ax.legend(fancybox=True, ncol=3)
                plt.ylim(ymin=ymin, ymax=ymax)
                plt.xlim(xmin=xmin, xmax=xmax)
                # ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))


class NLL(Plot):

    def __init__(self, subdir='nll', transform=True, dimensionless=False, points=300):
        self.subdir = subdir
        self.transform = transform
        self.dimensionless = dimensionless
        self.points = points

    def specify(self, config, spec, index):
        inputs = multidim_np(config, spec, 1, points=self.points)

        for coefficient in config['coefficients']:
            spec.add(inputs, [], ['run', 'plot', '--coefficient', coefficient, '--index', index, config['fn']])

    def write(self, config, plotter, args):
        super(NLL, self).write(config)
        data = fit_nll(config, self.transform, self.dimensionless)
        scan = CrossSectionScan(os.path.join(config['outdir'], 'cross_sections.npz'))

        for coefficient in args.coefficients:
            info = data[coefficient]
            for p in config['processes']:
                s0, s1, s2 = scan.construct(p, [coefficient])
                if not ((s1 > 1e-5) or (s2 > 1e-5)):
                    continue  # coefficient has no effect on any of the scaled processes
            x_label = '{} {}'.format(info['label'].replace('\ \mathrm{TeV}^{-2}', ''), info['units'])

            with plotter.saved_figure(
                    x_label,
                    '$-2\ \Delta\ \mathrm{ln}\ \mathrm{L}$' + (' (asimov data)' if config['asimov data'] else ''),
                    os.path.join(self.subdir, coefficient),
                    header=config['header']) as ax:
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
                    ax.plot([low, high], [1.0, 1.0], '--', label=r'68% CL' if (i == 0) else '', color='blue')
                for i, (low, high) in enumerate(info['two sigma']):
                    ax.plot([low, high], [3.84, 3.84], '-.', label=r'95% CL' if (i == 0) else '', color='#ff321a')

                ax.legend(loc='upper center')
                ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
                plt.ylim(ymin=0, ymax=12)
                plt.xlim(xmin=info['x'][info['y'] < 13].min(), xmax=info['x'][info['y'] < 13].max())
                if info['transformed']:
                    plt.xlim(xmin=0)


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


class TwoProcessCrossSectionSM(Plot):

    def __init__(self, subdir='.', signals=['ttW', 'ttZ'], theory_errors=None, tag=None, numpoints=500, chunksize=500, contours=True):
        self.subdir = subdir
        self.signals = signals
        self.theory_errors = theory_errors
        if tag:
            self.tag = tag
        else:
            self.tag = '-'.join(signals)
        self.numpoints = numpoints
        self.chunksize = chunksize
        self.contours = contours

    def specify(self, config, spec, index):
        inputs = multi_signal(self.signals, self.tag, spec, config)
        for signal in self.signals:
            inputs += max_likelihood_fit(signal, spec, config)
        if self.contours:
            inputs += multidim_grid(config, self.tag, self.numpoints, self.chunksize, spec)

        spec.add(inputs, [],  ['run', 'plot', '--index', index, config['fn']])

    def write(self, config, plotter, args):
        x = self.signals[0]
        y = self.signals[1]
        with plotter.saved_figure(
                label['sigma {}'.format(x)],
                label['sigma {}'.format(y)],
                self.tag,
                header=config['header']) as ax:
            handles, labels = two_signal_best_fit(config, ax, self.signals, self.theory_errors, self.tag, self.contours)

            data = root2array(os.path.join(config['outdir'], 'best-fit-{}.root'.format(x)))

            x_cross_section = plt.axvline(x=data['limit'][0] * nlo[x], color='black')
            x_error = ax.axvspan(
                data['limit'][1] * nlo[x],
                data['limit'][2] * nlo[x],
                alpha=0.5,
                color='#FA6900',
                linewidth=0.0
            )
            handles.append((x_cross_section, x_error))
            labels.append('{} 1D $\pm$ $1\sigma$'.format(label[x]))

            data = root2array(os.path.join(config['outdir'], 'best-fit-{}.root'.format(y)))

            y_cross_section = plt.axhline(y=data['limit'][0] * nlo[y], color='black')
            y_error = ax.axhspan(
                data['limit'][1] * nlo[y],
                data['limit'][2] * nlo[y],
                color='#69D2E7',
                alpha=0.5,
                linewidth=0.0
            )
            handles.append((y_cross_section, y_error))
            labels.append('{} 1D $\pm$ $1\sigma$'.format(label[y]))

            plt.legend(handles, labels)


class TwoProcessCrossSectionSMAndNP(Plot):

    def __init__(self,
             subdir='.',
             signals=['ttW', 'ttZ'],
             theory_errors=None,
             tag=None,
             transform=True,
             dimensionless=False,
             points=300):
        self.subdir = subdir
        self.signals = signals
        self.theory_errors = theory_errors
        if tag:
            self.tag = tag
        else:
            self.tag = '-'.join(signals)
        self.transform = transform
        self.dimensionless = dimensionless
        self.points = points

    def specify(self, config, spec, index):
        inputs = multi_signal(self.signals, self.tag, spec, config)
        inputs += multidim_np(config, spec, 1, points=self.points)
        inputs += fluctuate(config, spec)

        spec.add(inputs, [],  ['run', 'plot', '--index', index, config['fn']])

    def write(self, config, plotter, args):
        x_proc = self.signals[0]
        y_proc = self.signals[1]
        nll = fit_nll(config, self.transform, self.dimensionless)

        table = []
        scales = ['r_{}'.format(x) for x in config['processes']]
        for coefficient in args.coefficients:
            data = np.load(os.path.join(config['outdir'], 'fluctuations-{}.npy'.format(coefficient)))[()]
            if np.isnan(data['x_sec_{}'.format(x_proc)]).any() or np.isnan(data['x_sec_{}'.format(y_proc)]).any():
                print 'skipping coefficient {} with nan fluctuations'.format(coefficient)
                continue

            with plotter.saved_figure(
                    label['sigma {}'.format(x_proc)],
                    label['sigma {}'.format(y_proc)],
                    os.path.join(self.subdir, '{}_{}'.format(self.tag, coefficient)),
                    header=config['header']) as ax:
                handles, labels = two_signal_best_fit(config, ax, self.signals, self.theory_errors, self.tag, False)

                x = data['x_sec_{}'.format(x_proc)]
                y = data['x_sec_{}'.format(y_proc)]

                try:
                    kdehist = kde.kdehist2(x, y, [70, 70])
                    clevels = sorted(kde.confmap(kdehist[0], [.6827, .9545]))
                    contour = ax.contour(kdehist[1], kdehist[2], kdehist[0], clevels, colors=['#ff321a', 'blue'], linestyles=['-.', '--'])
                    for handle, l in zip(contour.collections[::-1], ['68% CL', '95% CL']):
                        handles.append(handle)
                        labels.append(l)
                except Exception as e:
                    print 'problem making contour for {}: {}'.format(coefficient, e)

                colors = ['black', 'gray']
                for (bf, _), color in zip(nll[coefficient]['best fit'], colors):
                    table.append([coefficient, '{:.2f}'.format(bf), '{:.2f}'.format(data[0][coefficient])] + ['{:.2f}'.format(data[0][i]) for i in scales])
                    point, = plt.plot(
                        data[0]['x_sec_{}'.format(x_proc)],
                        data[0]['x_sec_{}'.format(y_proc)],
                        color=color,
                        markeredgecolor=color,
                        mew=3,
                        markersize=17,
                        marker="x",
                        linestyle='None'
                    )
                    handles.append(point)

                    labels.append("Best fit\n{}".format(nll[coefficient]['label']))
                plt.legend(handles, labels, loc='upper right', fontsize=27)
                plt.ylim(ymin=y_min, ymax=y_max)
                plt.xlim(xmin=x_min, xmax=x_max)
                ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
                ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))

        print tabulate.tabulate(table, headers=['coefficient', 'bf', 'coefficient value'] + config['processes'])


def plot(args, config):

    plotter = Plotter(config)

    if not args.coefficients:
        args.coefficients = config['coefficients']

    if args.header:
        config['header'] = args.header

    if args.index is not None:
        config['plots'][args.index].write(config, plotter, args)
    else:
        for p in config['plots']:
            p.write(config, plotter, args)
