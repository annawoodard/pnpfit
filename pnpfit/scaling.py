from collections import defaultdict
import os

import numpy as np
from numpy.polynomial import Polynomial
import tabulate

from NPFitProduction.NPFitProduction.cross_sections import CrossSectionScan


def load(config):
    fn = os.path.join(config['outdir'], 'cross_sections.npz')
    scan = CrossSectionScan(fn)

    return scan.points, scan.cross_sections


def load_fitted_scan(config, fn='cross_sections.npz', maxpoints=None, dimension=None):
    fn = os.path.join(config['outdir'], fn)
    scan = CrossSectionScan(fn)

    scan.fit(maxpoints=maxpoints, dimensions=[dimension])

    return scan


def load_scales(config):
    scales = defaultdict(dict)
    fn = os.path.join(config['outdir'], 'cross_sections.npz')
    scan = CrossSectionScan(fn)

    for coefficients in scan.points:
        scan.fit(coefficients)
        for process in scan.points[coefficients]:
            if coefficients == 'sm':
                continue
            scales[coefficients[0]][process] = Polynomial(scan.fit_constants[coefficients][process])

    return scales


def dump_scales(args, config):
    scales = load_scales(config)

    np.save(os.path.join(config['outdir'], 'scales.npy'), scales)

    table = []
    for c in scales:
        for p in scales[c]:
            table.append([c, p] + list(scales[c][p].coef))

    with open(os.path.join(config['outdir'], 'scales.txt'), 'w') as f:
        f.write(tabulate.tabulate(table, headers=['Wilson coefficients', 'process', 's0', 's1', 's2']))
