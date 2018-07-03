# import NPFit.NPFit.plotting as plotting
# import NPFit.NPFit.tabulation as tabulation
# from NPFit.NPFit.parameters import conversion
import os

import pnpfit

scanpoints = 3000

data = os.path.join(pnpfit.__path__[0], 'data')

config = {
        'release': '/home/annawoodard/ttV/CMSSW_8_1_0',
        'cross sections': os.path.join(data, 'cross_sections/13-TeV/merged/merged.npz'),
        'outdir': os.path.abspath('results'),
        'coefficients': ['cuB'],
        # 'coefficients': ['cuW', 'cuB', 'cH', 'tc3G', 'c3G', 'cHu', 'c2G', 'cuG'],
        # 'plots': [
        #     plotting.NewPhysicsScaling([('ttW', 'x', 'blue'), ('ttZ', '+', '#2fd164'), ('ttH', 'o', '#ff321a')]),
        #     plotting.NLL(),
        #     plotting.TwoProcessCrossSectionSMAndNP(
        #         subdir='.',
        #         signals=['ttW', 'ttZ'],
        #         theory_errors={'ttW': (0.1173, 0.1316), 'ttZ': (0.1164, 0.10)}),
        #     plotting.TwoProcessCrossSectionSM(
        #         subdir='.',
        #         signals=['ttW', 'ttZ'],
        #         theory_errors={'ttW': (0.1173, 0.1316), 'ttZ': (0.1164, 0.10)},
        #         numpoints=500, chunksize=250, contours=True),
        #     plotting.FitQualityByDim(fit_dimensions=[[2], [8]], eval_dimensions=[2, 8]),
        #     plotting.FitQualityByPoints(dimensions=[(2, 'o', 'orchid'), (8, 's', 'navy')], points=range(1, 85, 1)),
        #     plotting.NewPhysicsScaling2D(subdir='scaling-2d-128d-fit', dimensions=[1, 2, 8], maxnll=70, points=scanpoints),
        #     plotting.NewPhysicsScaling2D(subdir='scaling-2d-128d-fit-frozen', dimensions=[1, 2, 8], maxnll=70,
        #         points=scanpoints, profile=False),
        #     plotting.NLL2D(draw='mesh', maxnll=70, points=scanpoints),
        #     ],
        # 'tables': [
        #     tabulation.CLIntervals(dimension=1),
        #     tabulation.CLIntervals(dimension=2),
        #     tabulation.CLIntervals(dimension=8),
        #     tabulation.CLIntervals(dimension=8, freeze=True)
        # ],
        'np chunksize': 100,
        'asimov data': False,  # Calculate expected values with MC data only (Asimov dataset), false for real data.
        'cards': {
            'ttW': os.path.join(data, 'cards/TOP-17-005/ttW.txt'),
            'ttZ': os.path.join(data, 'cards/TOP-17-005/ttZ.txt'),
            'ttV_np': os.path.join(data, 'cards/TOP-17-005/ttV_np.txt')
            },
        'luminosity': 36,
        'scale window': 10,  # maximum scaling of any scaled process at which to set the scan boundaries
        'scan window': {
            ('c2G', 'c3G'): [-14., 14., -13., 13.],
            ('c2G', 'cH'): [-13., 13., -41., 70.],
            ('c2G', 'cHu'): [-13., 13., -15., 15.],
            ('c2G', 'cuB'): [-13., 13., -13., 11.],
            ('c2G', 'cuG'): [-13., 13., -2.5, 2.],
            ('c2G', 'cuW'): [-13., 13., -14., 14.],
            ('c2G', 'tc3G'): [-13., 14., -2.3, 2.3],
            ('c3G', 'cH'): [-13., 13., -41., 70.],
            ('c3G', 'cHu'): [-13., 13., -29., 20.],
            ('c3G', 'cuB'): [-13., 13., -13., 13.],
            ('c3G', 'cuG'): [-13., 13., -2.5, 2.5],
            ('c3G', 'cuW'): [-13., 13., -14., 14.],
            ('c3G', 'tc3G'): [-13., 13., -2., 2.7],
            ('cH', 'cHu'): [-41., 70., -29.1, 20.],
            ('cH', 'cuB'): [-41., 69., -13.0, 13.],
            ('cH', 'cuG'): [-41., 70., -2.5, 2.],
            ('cH', 'cuW'): [-41., 70., -14., 14.2],
            ('cH', 'tc3G'): [-41., 69., -2.3, 2.7],
            ('cHu', 'cuB'): [-29., 20., -11., 13.],
            ('cHu', 'cuG'): [-29., 21., -2.5, 2.5],
            ('cHu', 'cuW'): [-28., 20., -14., 14.],
            ('cHu', 'tc3G'):[ -29., 20., -2., 2.7],
            ('cuB', 'cuG'): [-11., 13., -2.5, 2.],
            ('cuB', 'cuW'): [-13., 11., -14., 14.],
            ('cuB', 'tc3G'):[-13., 13., -2., 2.7],
            ('cuG', 'cuW'): [-2., 3., -14.2, 14.2],
            ('cuG', 'tc3G'): [-2., 2., -2.3, 2.7],
            ('cuW', 'tc3G'): [-14., 14., -2., 2.7]
        },
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
