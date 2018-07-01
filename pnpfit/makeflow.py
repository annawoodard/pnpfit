import glob
import logging
import os
import re
import shlex
import shutil
import subprocess

import numpy as np

from NPFit.NPFit.actionable import annotate
from NPFitProduction.NPFitProduction.utils import sorted_combos


def max_likelihood_fit(analysis, spec, config):
    workspace = os.path.join(config['outdir'], 'workspaces', '{}.root'.format(analysis))
    card = os.path.join(config['outdir'], '{}.txt'.format(analysis))
    spec.add(card, workspace, ['text2workspace.py', card, '-o', workspace])
    best_fit = os.path.join(config['outdir'], 'best-fit-{}.root'.format(analysis))
    fit_result = os.path.join(config['outdir'], 'fit-result-{}.root'.format(analysis))
    cmd = 'combine -M MaxLikelihoodFit {a} >& {a}.fit.log'.format(a=card)
    outputs = {
        'higgsCombineTest.MaxLikelihoodFit.mH120.root': best_fit,
        'fitDiagnostics.root': fit_result
    }
    spec.add(workspace, outputs, cmd)

    return [best_fit, fit_result]


def multi_signal(signals, tag, spec, config):
    workspace = os.path.join(config['outdir'], 'workspaces', '{}.root'.format(tag))
    card = os.path.join(config['outdir'], '{}.txt'.format(tag))
    cmd = [
        'text2workspace.py', card,
        '-P', 'HiggsAnalysis.CombinedLimit.PhysicsModel:multiSignalModel',
        '-o', workspace
    ] + ['--PO map=.*/{signal}:r_{signal}[1,0,4]'.format(signal=s) for s in signals]
    spec.add([card], workspace, cmd)

    best_fit = os.path.join(config['outdir'], 'best-fit-{}.root'.format(tag))
    fit_result = os.path.join(config['outdir'], 'fit-result-{}.root'.format(tag))
    outputs = {
        'higgsCombineTest.MultiDimFit.mH120.root': best_fit,
    }
    cmd = 'combine -M MultiDimFit {} --autoBoundsPOIs=* --saveFitResult --algo=cross >& {}.fit.log'.format(workspace, card)
    spec.add(workspace, outputs, cmd)

    return [best_fit]


def multidim_grid(config, tag, points, chunksize, spec):
    workspace = os.path.join(config['outdir'], 'workspaces', '{}.root'.format(tag))
    lowers = np.arange(1, points - 1, chunksize)
    uppers = np.arange(chunksize, points, chunksize)
    scans = []
    for index, (first, last) in enumerate(zip(lowers, uppers)):
        filename = 'higgsCombine_{}_{}.MultiDimFit.mH120.root'.format(tag, index)
        scan = os.path.join(config['outdir'], 'scans', filename)
        scans.append(scan)

        cmd = [
            'combine',
            '-M', 'MultiDimFit',
            '--saveFitResult',
            workspace,
            '--algo=grid',
            '--points={}'.format(points),
            '-n', '_{}_{}'.format(tag, index),
            '--firstPoint {}'.format(first),
            '--lastPoint {}'.format(last),
            '--autoBoundsPOIs=*'
        ]

        spec.add(workspace, {filename: scan}, cmd)

    outfile = os.path.join(config['outdir'], 'scans', '{}.total.root'.format(tag))
    spec.add(scans, outfile, ['hadd', '-f', outfile] + scans)

    return [outfile]


def multidim_np(config, spec, dimension, points=None, cl=None, freeze=True):
    outfiles = []
    freeze = ['--freeze'] if freeze is True else []
    def make_workspace(coefficients):
        workspace = os.path.join(config['outdir'], 'workspaces', '{}.root'.format('_'.join(coefficients)))
        cmd = [
            'text2workspace.py', os.path.join(config['outdir'], 'ttV_np.txt'),
            '-P', 'NPFit.NPFit.models:eft',
            '--PO', 'scan={}'.format(os.path.join(config['outdir'], 'cross_sections.npz')),
            ' '.join(['--PO process={}'.format(x) for x in config['processes']]),
            ' '.join(['--PO poi={}'.format(x) for x in coefficients]),
            '-o', workspace
        ]
        spec.add(['cross_sections.npz'], workspace, cmd)

        return workspace
    outfiles += [make_workspace(config['coefficients'])]
    for coefficients in sorted_combos(config['coefficients'], dimension):
        workspace = make_workspace(coefficients)
        if dimension == 1:
            label = coefficients[0]
        else:
            label = '{}{}'.format('_'.join(coefficients), '_frozen' if freeze else '')

        best_fit = os.path.join(config['outdir'], 'best-fit-{}.root'.format(label))
        fit_result = os.path.join(config['outdir'], 'fit-result-{}.root'.format(label))
        cmd = ['run', 'combine'] + freeze + list(coefficients) + [config['fn']]
        if dimension == 1 and cl is None:
            spec.add([workspace], [best_fit, fit_result], cmd)
            outfiles += [best_fit, fit_result]
        if points is None and cl is None:
            spec.add([workspace], [best_fit], cmd)
            outfiles += [best_fit]
        elif cl is not None:
            for level in cl:
                outfile = os.path.join(config['outdir'], 'cl_intervals/{}-{}.root'.format(label, level))
                cmd = ['run', 'combine'] + freeze + list(coefficients) + ['--cl', str(level), config['fn']]
                spec.add(['cross_sections.npz', workspace], outfile, cmd)
                outfiles += [outfile]
        else:
            cmd = ['run', 'combine', '--snapshot'] + freeze + list(coefficients) + [config['fn']]
            snapshot = os.path.join(config['outdir'], 'snapshots', '{}.root'.format(label))
            spec.add([workspace], snapshot, cmd)

            scans = []
            for index in range(int(np.ceil(points / config['np chunksize']))):
                scan = os.path.join(config['outdir'], 'scans', '{}_{}.root'.format(label, index))
                scans.append(scan)
                cmd = ['run', 'combine'] + freeze + list(coefficients) + ['-i', str(index), '-p', str(points), config['fn']]

                spec.add(['cross_sections.npz', snapshot], scan, cmd)

            total = os.path.join(config['outdir'], 'scans', '{}.total.root'.format(label))
            spec.add(scans, total, ['hadd', '-f', total] + scans)
            outfiles += [total]

    return list(set(outfiles))


def fluctuate(config, spec):
    outfiles = []
    for coefficients in sorted_combos(config['coefficients'], 1):
        label = '_'.join(coefficients)
        fit_result = os.path.join(config['outdir'], 'fit-result-{}.root'.format(label))
        cmd = ['run', 'fluctuate', label, config['fluctuations'], config['fn']]
        outfile = os.path.join(config['outdir'], 'fluctuations-{}.npy'.format(label))
        spec.add([fit_result], outfile, cmd)
        outfiles += [outfile]

    return outfiles


def make(args, config):
    def cardify(name):
        return os.path.join(config['outdir'], '{}.txt'.format(name))

    if os.path.isfile(os.path.join(config['outdir'], 'config.py')):
        raise ValueError('refusing to overwrite outdir {}'.format(config['outdir']))

    shutil.copy(args.config, config['outdir'])

    prepare_cards(args, config, cardify)

    makefile = os.path.join(config['outdir'], 'Makeflow')
    logging.info('writing Makeflow file to {}'.format(config['outdir']))

    annotate(args, config)

    spec = MakeflowSpecification(config['fn'])

    files = sum([glob.glob(os.path.join(indir, '*.root')) for indir in config['indirs']], [])
    for f in files:
        outputs = os.path.join('cross_sections', os.path.basename(f).replace('.root', '.npz'))
        spec.add([], outputs, ['run', '--parse', f, config['fn']])

    inputs = [os.path.join('cross_sections', os.path.basename(f).replace('.root', '.npz')) for f in files]
    if 'indirs' in config:
        for indir in config['indirs']:
            for root, _, filenames in os.walk(indir):
                inputs += [os.path.join(root, fn) for fn in filenames if fn.endswith('.npz')]
    spec.add(inputs, 'cross_sections.npz', ['LOCAL', 'run', 'concatenate', config['fn']])

    for index, plot in enumerate(config.get('plots', [])):
        plot.specify(config, spec, index)

    for index, table in enumerate(config.get('tables', [])):
        table.specify(config, spec, index)

    spec.dump(makefile)
