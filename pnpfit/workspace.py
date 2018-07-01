import os
from pnpfit.util import cmssw_call, sorted_combos

def make_np_workspaces(futures, config, dimension=1):
    workspaces = {}

    if not os.path.isdir(os.path.join(config['outdir'], 'workspaces')):
        os.makedirs(os.path.join(config['outdir'], 'workspaces'))

    def make_workspace(coefficients):
        workspace = os.path.join(config['outdir'], 'workspaces', '{}.root'.format('_'.join(coefficients)))
        cmd = [
            'text2workspace.py', os.path.join(config['outdir'], 'ttV_np.txt'),
            '-P', 'NPFit.NPFit.models:eft',
            '--PO', 'scan={}'.format(config['cross sections']),
            ' '.join(['--PO process={}'.format(x) for x in config['processes']]),
            ' '.join(['--PO poi={}'.format(x) for x in coefficients]),
            '-o', workspace
        ]
        return cmssw_call(cmd, outputs=[workspace]).outputs[0]

    workspaces[tuple(config['coefficients'])] = make_workspace(config['coefficients'])
    for coefficients in sorted_combos(config['coefficients'], dimension):
        workspaces[coefficients] = make_workspace(coefficients)

    futures['workspaces'] = workspaces
