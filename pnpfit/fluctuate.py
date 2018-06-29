import os

import numpy as np
import ROOT

from NPFit.NPFit.parameters import nlo

from NPFitProduction.NPFitProduction.cross_sections import CrossSectionScan


def fluctuate(args, config):
    ROOT.RooRandom.randomGenerator().SetSeed(0)

    coefficient = args.coefficient
    scan = CrossSectionScan(os.path.join(config['outdir'], 'cross_sections.npz'))

    file = ROOT.TFile.Open(os.path.join(config['outdir'], 'fit-result-{}.root'.format(coefficient)))
    fit = file.Get('fit_mdf')

    dtype = [('x_sec_{}'.format(process), 'f8') for process in config['processes']]
    dtype += [('r_{}'.format(process), 'f8') for process in config['processes']]
    dtype += [(coefficient, 'f8')] + [(s, 'f8') for s in config['systematics'].keys()]
    data = np.empty((int(args.perturbations) + 1,), dtype=dtype)

    pars = fit.floatParsFinal()
    for theta in config['systematics'].keys() + [args.coefficient]:
        data[0][theta] = float(pars.selectByName(theta)[0].getVal())

    def get_cross_sections(pars):
        cross_sections = {}

        for process in config['processes']:
            cross_sections[process] = ROOT.ProcessNormalization('xsec_{}'.format(process), '', nlo[process])
            for systematic, info in config['systematics'].items():
                if process in info['kappa']:
                    if not info['distribution'] == 'lnN':
                        raise NotImplementedError
                    if info['kappa'][process]['+'] == info['kappa'][process]['-']:
                        cross_sections[process].addLogNormal(info['kappa'][process]['+'], pars.selectByName(systematic)[0])
                    else:
                        cross_sections[process].addAsymmLogNormal(
                            info['kappa'][process]['-'],
                            info['kappa'][process]['+'],
                            pars.selectByName(systematic)[0]
                        )

        return cross_sections

    cross_sections = get_cross_sections(pars)
    for process in config['processes']:
        data[0]['x_sec_{}'.format(process)] = cross_sections[process].getVal() * scan.evaluate(tuple([coefficient]), np.array([[data[coefficient][0]]]), process)

    pars = fit.randomizePars()
    cross_sections = get_cross_sections(pars)
    for i in range(1, int(args.perturbations)):
        pars = fit.randomizePars()
        for par in config['systematics'].keys() + [coefficient]:
            data[i][par] = pars.selectByName(par)[0].getVal()
        for process in config['processes']:
            scale = scan.evaluate(tuple([coefficient]), np.array([[data[i][coefficient]]]), process)
            data[i]['x_sec_{}'.format(process)] = cross_sections[process].getVal() * scale

    for process in config['processes']:
        print(data[coefficient])
        print(data[coefficient].shape)
        print(np.array(data[coefficient]))
        print(np.array(data[coefficient]).shape)
        data['r_{}'.format(process)] = scan.evaluate(tuple([coefficient]), data[coefficient].reshape((len(data[coefficient]), 1)), process)

    np.save(os.path.join(config['outdir'], 'fluctuations-{}.npy'.format(args.coefficient)), data)
