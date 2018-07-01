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

