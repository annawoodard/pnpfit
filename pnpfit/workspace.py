import os
from pnpfit.util import cmssw_call


def make_np_workspace(coefficients, outdir, cross_sections, processes):
    if not isinstance(coefficients, list):
        coefficients = [coefficients]
    if not os.path.isdir(outdir):
        os.makedirs(outdir)
    workspace = os.path.join(outdir, 'workspaces', '{}.root'.format('_'.join(coefficients)))
    command = [
        'text2workspace.py', os.path.join(outdir, 'ttV_np.txt'),
        '-P', 'NPFit.NPFit.models:eft',
        '--PO', 'scan={}'.format(cross_sections),
        ' '.join(['--PO process={}'.format(x) for x in processes]),
        ' '.join(['--PO poi={}'.format(x) for x in coefficients]),
        '-o', workspace
    ]

    return cmssw_call(command, outputs=[workspace]).outputs[0]

