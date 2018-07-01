import os

import numpy as np

# from pnpfit.util import call, cmssw_call, sorted_combos
import pnpfit.data
from pnpfit.util import cmssw_call, sorted_combos
from pnpfit.cross_sections import CrossSectionScan

def combine(futures, config, dimension, points=None, cl=None, freeze=True, snapshot=False, index=None):
    for coefficients in sorted_combos(config['coefficients'], dimension):
        if len(coefficients) == 1:
            label = coefficients[0]
        else:
            label = '{}{}'.format('_'.join(coefficients), '_frozen' if freeze else '')

        all_coefficients = tuple(sorted(config['coefficients']))
        scan = CrossSectionScan([config['cross sections']])
        mins = np.amin(scan.points[all_coefficients][config['processes'][-1]], axis=0)
        maxes = np.amax(scan.points[all_coefficients][config['processes'][-1]], axis=0)

        def get_command(postfix, wspace=None):
            cmd = ['combine', '-M', 'MultiDimFit']
            if len(coefficients) == 1:
                if wspace is None:
                    wspace = os.path.join(config['outdir'], 'workspaces', '{}.root'.format(coefficients[0]))
                cmd += [
                    wspace,
                    '--setParameters {}=0.0'.format(coefficients[0]),
                    '--autoRange=20'
                ]
            else:
                if wspace is None:
                    wspace = os.path.join(config['outdir'], 'workspaces', '{}.root'.format('_'.join(config['coefficients'])))
                cmd += [
                    wspace,
                    '--setParameters', '{}'.format(','.join(['{}=0.0'.format(x) for x in config['coefficients']])),
                    '--floatOtherPOIs={}'.format(int(not freeze)),
                    '--robustFit=1',
                    '--setRobustFitTolerance=0.001',
                    '--cminApproxPreFitTolerance=0.1',
                    '--cminPreScan'
                ] + ['-P {}'.format(p) for p in coefficients]
                ranges = []
                if tuple(coefficients) in config.get('scan window', []):
                    x = coefficients[0]
                    y = coefficients[1]
                    xmin, xmax, ymin, ymax = [np.array(i) for i in config['scan window'][tuple(coefficients)]]
                    ranges += ['{c}={low},{high}'.format(c=x, low=xmin / conversion[x], high=xmax / conversion[x])]
                    ranges += ['{c}={low},{high}'.format(c=y, low=ymin / conversion[y], high=ymax / conversion[y])]
                for c, low, high in zip(all_coefficients, mins, maxes):
                    if (tuple(coefficients) not in config.get('scan window', [])) \
                            or (c not in coefficients) \
                            or (len(coefficients) != 2):
                        ranges += ['{c}={low},{high}'.format(c=c, low=low * 10., high=high * 10.)]
                cmd += [
                    '--setParameterRanges', ':'.join(ranges)
                ]
            if config['asimov data']:
                cmd += ['-t', '-1']

            cmd += postfix
            return cmd

        workspace = futures['workspaces'][coefficients]
        if snapshot:
            snapshot = os.path.join(config['outdir'], 'snapshots', '{}.root'.format(label))
            command = get_command(['--saveWorkspace', '--saveInactivePOI=1'])
            futures['snapshots'][coefficients] = cmssw_call(
                command,
                inputs=[workspace],
                outputs=[snapshot],
                rename=['higgsCombineTest.MultiDimFit.mH120.root->{}'.format(snapshot)]
            ).outputs[0]

        elif index is None:
            if len(coefficients) == 1 and cl is None:
                fit_result_path = os.path.join(config['outdir'], 'fit-result-{}.root'.format(label))
                best_fit_path = os.path.join(config['outdir'], 'best-fit-{}.root'.format(label))
                command = get_command(['--saveFitResult', '--algo=singles'])
                rename = [
                    'multidimfit.root->{}'.format(fit_result_path),
                    'higgsCombineTest.MultiDimFit.mH120.root->{}'.format(best_fit_path)
                ]
                fit_result, best_fit = cmssw_call(
                    command,
                    inputs=[workspace],
                    outputs=[fit_result_path, best_fit_path],
                    rename=rename
                ).outputs
                futures['fit result'][coefficients], futures['best fit'][coefficients] = cmssw_call(
                    [os.path.join(pnpfit.data.__path__[0], 'root2array.py'), fit_result_path, best_fit_path],
                    inputs=[workspace, fit_result, best_fit],
                    outputs=[fit_result_path.replace('.root', '.npy'), best_fit_path.replace('.root', '.npy')],
                ).outputs
            elif cl is None:
                best_fit = os.path.join(config['outdir'], 'best-fit-{}.root'.format(label))
                futures['best fit'][coefficients] = cmssw_call(
                    command,
                    rename=['higgsCombineTest.MultiDimFit.mH120.root->{}'.format(best_fit)],
                    inputs=[workspace],
                    outputs=[best_fit]
                ).outputs[0]
            else:
                interval = os.path.join(config['outdir'], 'cl_intervals/{}-{}.root'.format(label, cl))
                command = get_command([
                    '--algo=cross',
                    '--stepSize=0.01',
                    '--cl={}'.format(cl),
                    '; mv higgsCombineTest.MultiDimFit.mH120.root {}'.format(interval)
                ])
                futures['cl interval'][coefficients][cl] = cmssw_call(command, inputs=[workspace], outputs=[interval]).outputs[0]
        else:
            lowers = np.arange(1, points, config['np chunksize'])
            uppers = np.arange(config['np chunksize'], points + config['np chunksize'], config['np chunksize'])
            first, last = zip(lowers, uppers)[index]
            scan_chunk = os.path.join(config['outdir'], 'scans', '{}_{}.root'.format(label, index))
            postfix = [
                '--algo=grid',
                '--points={}'.format(points),
                '--firstPoint={}'.format(first),
                '--lastPoint={}'.format(last),
            ]
            if len(coefficients) == 1:
                command = get_command(postfix + ['; mv higgsCombineTest.MultiDimFit.mH120.root {}'.format(scan_chunk)])
            else:
                wspace = os.path.join(config['outdir'], 'snapshots', '{}.root'.format(label))
                postfix += ['-w', 'w', '--snapshotName', 'MultiDimFit', '; mv higgsCombineTest.MultiDimFit.mH120.root {}'.format(scan_chunk)]
                command = get_command(postfix, wspace)
            futures['scans'][coefficients][index] = cmssw_call(command, inputs=[workspace], outputs=[scan_chunk]).outputs[0]



def fit(futures, config, dimension, points=None, cl=None, freeze=True):
    for coefficients in sorted_combos(config['coefficients'], dimension):
        if (dimension == 1 and cl is None) or (points is None and cl is None):
            combine(futures, config, freeze, coefficients, cl, points)
        elif cl is not None:
            for level in cl:
                combine(config, freeze, coefficients, cl, points)
        else:
            combine(futures, config, freeze, coefficients, cl, points, snapshot=True)

            scans = []
            for index in range(int(np.ceil(points / config['np chunksize']))):
                combine(futures, config, freeze, coefficients, cl, points, index=index)

            if dimension == 1:
                label = coefficients[0]
            else:
                label = '{}{}'.format('_'.join(coefficients), '_frozen' if freeze else '')

            total = os.path.join(config['outdir'], 'scans', '{}.total.root'.format(label))
            command = ['hadd', '-f', total] + [f.filepath for f in futures['scans'][coefficients].values()]
            scans = futures['scans'][coefficients].values()
            futures['scan total'][coefficients] = cmssw_call(command, inputs=scans, outputs=[total]).outputs[0]
