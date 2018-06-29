import os

import NPFit.NPFit.plotting as plotting
import NPFit.NPFit.tabulation as tabulation
from NPFit.NPFit.parameters import conversion

config = {
        'indirs': [
            os.path.join(os.environ['CMSSW_BASE'], 'src/NPFit/NPFit/data/cross_sections/13-TeV/merged')
            ],
        'outdir': '~/www/ttV/1/',  # Output directory; iterate the version each time you make changes
        'shared-fs': ['/afs', '/hadoop'],  # Declare filesystems the batch system can access-- files will not be copied (faster)
        'coefficients': ['cuW', 'cuB', 'cH', 'tc3G', 'c3G', 'cHu', 'c2G', 'cuG'],
        'batch type': 'condor',  # batch type for makeflow, can be local, wq, condor, sge, torque, moab, slurm, chirp, amazon, dryrun
        'plots': [
            plotting.NewPhysicsScaling([('ttW', 'x', 'blue'), ('ttZ', '+', '#2fd164'), ('ttH', 'o', '#ff321a')]),
            plotting.NLL(),
            plotting.TwoProcessCrossSectionSM(
                subdir='.',
                signals=['ttW', 'ttZ'],
                theory_errors={'ttW': (0.1173, 0.1316), 'ttZ': (0.1164, 0.10)},
                numpoints=500, chunksize=250, contours=True),
            plotting.TwoProcessCrossSectionSMAndNP(
                subdir='.',
                signals=['ttW', 'ttZ'],
                theory_errors={'ttW': (0.1173, 0.1316), 'ttZ': (0.1164, 0.10)}),
            ],
        'tables': [
            tabulation.CLIntervals(dimension=1),
            ],
        'np chunksize': 100,
        'asimov data': False,  # Calculate expected values with MC data only (Asimov dataset), false for real data.
        'cards': {
            '2l': 'data/cards/TOP-17-005/2l',
            '3l': 'data/cards/TOP-17-005/3l',
            '4l': 'data/cards/TOP-17-005/4l.txt'
            },
        'luminosity': 36,
        'scale window': 10,  # maximum scaling of any scaled process at which to set the scan boundaries
        'processes': ['ttH', 'ttZ', 'ttW'],  # processes to scale
        'fluctuations': 10000,
        'header': 'preliminary',
        'systematics': {  # below, list any additional (NP-specific, beyond what is in `cards`) systematics to apply
            'PDF_gg': {
                'kappa': {  # https://arxiv.org/pdf/1610.07922.pdf page 160
                    'ttZ': {'-': 1.028, '+': 1.028},
                    'ttH': {'-': 1.03, '+': 1.03}
                    },
                'distribution': 'lnN'
                },
            'PDF_qq': {
                'kappa': {'ttW': {'-': 1.0205, '+': 1.0205}},
                'distribution': 'lnN'
                },
            'Q2_ttH': {
                'kappa': {'ttH': {'-': 1.092, '+': 1.058}},
                'distribution': 'lnN'
                },
            'Q2_ttZ': {
                'kappa': {'ttZ': {'-': 1.113, '+': 1.096}},
                'distribution': 'lnN'
                },
            'Q2_ttW': {
                'kappa': {'ttW': {'-': 1.1155, '+': 1.13}},
                'distribution': 'lnN'
                }
            },
        'label': 'ttV_FTW',  # label to use for batch submission: no need to change this between runs
}
