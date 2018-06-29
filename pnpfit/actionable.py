import glob
import logging
import numpy as np
import os
import shlex
import shutil
import subprocess
import sys

from NPFitProduction.NPFitProduction.cross_sections import CrossSectionScan
from NPFit.NPFit.parameters import conversion

def annotate(args, config):
    """Annotate the output directory with a README

    The README includes instructions to reproduce the current version of
    the code. Any unstaged code changes will be saved as a git patch.
    """
    start = os.getcwd()
    os.chdir(os.path.dirname(__file__))
    head = subprocess.check_output(shlex.split('git rev-parse --short HEAD')).strip()
    diff = subprocess.check_output(shlex.split('git diff'))
    os.chdir(start)

    shared_filesystem = []
    if 'shared-fs' in config:
        for directory in config['shared-fs']:
            if not directory.startswith('/'):
                raise Exception('absolute path required for shared filesystems: {}'.format(directory))
            shared_filesystem += ["--shared-fs '/{}'".format(directory.split('/')[1])]

    info = """
    # to run, go to your output directory:
    cd {outdir}

    # if you are using a batch queue, start a factory to submit workers
    # this does not need to be run every time; it can be left running in the background and will only
    # submit workers as needed
    nohup work_queue_factory -T {batch_type} -M {label} -C {factory} >& factory.log &

    # execute the makeflow
    makeflow -T wq -M {label} {shared}

    # alternatively, if you do not have much work to do, it may be faster to run locally instead:
    makeflow -T local

    # to reproduce the code:
    cd {code_dir}
    git checkout {head}
    """.format(
        batch_type=config['batch type'],
        outdir=config['outdir'],
        label=config['label'],
        factory=os.path.join(os.environ['LOCALRT'], 'src', 'NPFit', 'NPFit', 'data', 'factory.json'),
        shared=' '.join(shared_filesystem),
        code_dir=os.path.dirname(__file__),
        head=head
    )

    if diff:
        with open(os.path.join(config['outdir'], 'patch.diff'), 'w') as f:
            f.write(diff)
        info += 'git apply {}\n'.format(os.path.join(config['outdir'], 'patch.diff'))

    with open(os.path.join(config['outdir'], 'README.txt'), 'w') as f:
        f.write(info)
    logging.info(info)


def parse(args, config):
    import DataFormats.FWLite

    result = CrossSectionScan()

    def get_collection(run, ctype, label):
        handle = DataFormats.FWLite.Handle(ctype)
        try:
            run.getByLabel(label, handle)
        except:
            raise

        return handle.product()

    logging.info('parsing {}'.format(args.file))

    for run in DataFormats.FWLite.Runs(args.file):
        cross_section = get_collection(run, 'LHERunInfoProduct', 'externalLHEProducer::LHE').heprup().XSECUP[0]
        coefficients = get_collection(run, 'vector<string>', 'annotator:wilsonCoefficients:LHE')
        process = str(get_collection(run, 'std::string', 'annotator:process:LHE'))
        point = np.array(get_collection(run, 'vector<double>', 'annotator:point:LHE'))

        result.add(point, cross_section, process, coefficients)

    outfile = os.path.join(config['outdir'], 'cross_sections', os.path.basename(args.file).replace('.root', '.npz'))
    result.dump(outfile)


def concatenate(args, config):
    if args.files is not None:
        files = sum([glob.glob(x) for x in args.files], [])
    else:
        files = glob.glob(os.path.join(config['outdir'], 'cross_sections', '*.npz'))
        if 'indirs' in config:
            for indir in config['indirs']:
                for root, _, filenames in os.walk(indir):
                    files += [os.path.join(root, fn) for fn in filenames if fn.endswith('.npz')]

    result = CrossSectionScan(files)
    result.fit()

    outfile = os.path.join(config['outdir'], args.output)
    result.dump(outfile)


def combine(args, config):
    if len(args.coefficients) == 1:
        label = args.coefficients[0]
    else:
        label = '{}{}'.format('_'.join(args.coefficients), '_frozen' if args.freeze else '')

    all_coefficients = tuple(sorted(config['coefficients']))
    scan = CrossSectionScan([os.path.join(config['outdir'], 'cross_sections.npz')])
    mins = np.amin(scan.points[all_coefficients][config['processes'][-1]], axis=0)
    maxes = np.amax(scan.points[all_coefficients][config['processes'][-1]], axis=0)

    def call_combine(postfix, wspace=None):
        cmd = ['combine', '-M', 'MultiDimFit']
        if len(args.coefficients) == 1:
            if wspace is None:
                wspace = os.path.join(config['outdir'], 'workspaces', '{}.root'.format(args.coefficients[0]))
            cmd += [
                wspace,
                '--setParameters {}=0.0'.format(args.coefficients[0]),
                '--autoRange=20'
            ]
        else:
            if wspace is None:
                wspace = os.path.join(config['outdir'], 'workspaces', '{}.root'.format('_'.join(config['coefficients'])))
            cmd += [
                wspace,
                '--setParameters', '{}'.format(','.join(['{}=0.0'.format(x) for x in config['coefficients']])),
                '--floatOtherPOIs={}'.format(int(not args.freeze)),
                '--robustFit=1',
                '--setRobustFitTolerance=0.001',
                '--cminApproxPreFitTolerance=0.1',
                '--cminPreScan'
            ] + ['-P {}'.format(p) for p in args.coefficients]
            ranges = []
            if tuple(args.coefficients) in config.get('scan window', []):
                x = args.coefficients[0]
                y = args.coefficients[1]
                xmin, xmax, ymin, ymax = [np.array(i) for i in config['scan window'][tuple(args.coefficients)]]
                ranges += ['{c}={low},{high}'.format(c=x, low=xmin / conversion[x], high=xmax / conversion[x])]
                ranges += ['{c}={low},{high}'.format(c=y, low=ymin / conversion[y], high=ymax / conversion[y])]
            for c, low, high in zip(all_coefficients, mins, maxes):
                if (tuple(args.coefficients) not in config.get('scan window', [])) \
                        or (c not in args.coefficients) \
                        or (len(args.coefficients) != 2):
                    ranges += ['{c}={low},{high}'.format(c=c, low=low * 10., high=high * 10.)]
            cmd += [
                # '--autoBoundsPOIs=*', '--autoMaxPOIs=*',# , '--verbose=1'
                '--setParameterRanges', ':'.join(ranges)
            ]
        if config['asimov data']:
            cmd += ['-t', '-1']

        cmd += postfix

        output = subprocess.check_output(' '.join(cmd), shell=True)
        if "MultiDimFit failed" in output:
            sys.exit(99)

    if args.snapshot:
        call_combine(['--saveWorkspace', '--saveInactivePOI=1'])
        shutil.move(
            'higgsCombineTest.MultiDimFit.mH120.root',
            os.path.join(config['outdir'], 'snapshots', '{}.root'.format(label)))
    elif args.index is None:
        if len(args.coefficients) == 1 and args.cl is None:
            call_combine(['--saveFitResult', '--algo=singles'])
            shutil.move(
                'multidimfit.root',
                os.path.join(config['outdir'], 'fit-result-{}.root'.format(label)))
            shutil.move(
                'higgsCombineTest.MultiDimFit.mH120.root',
                os.path.join(config['outdir'], 'best-fit-{}.root'.format(label)))
        elif args.cl is None:
            call_combine(['--algo=cross'])
            shutil.move(
                'higgsCombineTest.MultiDimFit.mH120.root',
                os.path.join(config['outdir'], 'best-fit-{}.root'.format(label)))
        else:
            call_combine(['--algo=cross', '--stepSize=0.01', '--cl={}'.format(args.cl)])
            shutil.move(
                'higgsCombineTest.MultiDimFit.mH120.root',
                os.path.join(config['outdir'], 'cl_intervals/{}-{}.root'.format(label, args.cl)))
    else:
        lowers = np.arange(1, args.points, config['np chunksize'])
        uppers = np.arange(config['np chunksize'], args.points + config['np chunksize'], config['np chunksize'])
        first, last = zip(lowers, uppers)[args.index]
        postfix = [
            '--algo=grid',
            '--points={}'.format(args.points),
            '--firstPoint={}'.format(first),
            '--lastPoint={}'.format(last),
        ]
        if len(args.coefficients) == 1:
            call_combine(postfix)
        else:
            wspace = os.path.join(config['outdir'], 'snapshots', '{}.root'.format(label))
            postfix += ['-w', 'w', '--snapshotName', 'MultiDimFit']
            call_combine(postfix, wspace)

        shutil.move(
            'higgsCombineTest.MultiDimFit.mH120.root',
            os.path.join(config['outdir'], 'scans', '{}_{}.root'.format(label, args.index)))
