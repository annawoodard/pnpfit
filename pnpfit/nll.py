import os

import numpy as np
from root_numpy import root2array
import scipy.signal
import tabulate

from NPFit.NPFit import line
from NPFit.NPFit.parameters import label, conversion

from NPFitProduction.NPFitProduction.cross_sections import CrossSectionScan


def fit_nll(config, transform=False, dimensionless=True):
    """Note that the best fit is not straightforward with multiple minima, see:
    https://hypernews.cern.ch/HyperNews/CMS/get/statistics/551/1.html
    """
    # TODO only run this once after running combine instead of every time I plot
    # TODO change to returning a dict with dimensionless and transformed

    scan = CrossSectionScan(os.path.join(config['outdir'], 'cross_sections.npz'))
    res = {}
    for coefficient in config['coefficients']:
        res[coefficient] = {
            'label': '',
            'best fit': [],
            'one sigma': [],
            'two sigma': [],
            'transformed': False
        }

        data = root2array(os.path.join(config['outdir'], 'scans', '{}.total.root'.format(coefficient)))
        # make sure min point is at 0 (combine might have chosen wrong best
        # fit for offset)
        data['deltaNLL'] -= data['deltaNLL'].min()
        max_nll = (14. / 2)
        # max_nll = (30. / 2)
        if data['deltaNLL'].max() > max_nll:
            while len(data[data['deltaNLL'] < max_nll][coefficient]) < 10:
                max_nll += 1
        data = data[data['deltaNLL'] < max_nll]
        _, unique = np.unique(data[coefficient], return_index=True)

        conversion_factor = 1.
        if not dimensionless:
            if coefficient not in conversion:
                print 'no conversion available for {}; reporting dimensionless value'.format(coefficient)
            else:
                conversion_factor = conversion[coefficient]

        x = data[unique][coefficient] * conversion_factor
        y = 2 * data[unique]['deltaNLL']

        if len(y) == 0:
            print 'skipping {}: no nll points found'.format(coefficient)
            continue

        minima = scipy.signal.argrelmin(y, order=5)
        threshold = (y[minima] - min(y)) < 0.1
        res[coefficient]['conversion'] = conversion_factor
        res[coefficient]['units'] = '' if conversion_factor == 1. else '$\ [\mathrm{TeV}^{-2}]$'
        if transform and (len(x[minima][threshold]) == 2 or config['asimov data']):
            xi = np.linspace(x.min(), x.max(), 10000)
            total = 0
            for process in config['processes']:
                total += scan.evaluate(coefficient, xi.reshape((len(xi), 1)), process)
            offset = xi[total.argmin()] * conversion_factor

            def transform(i):
                return np.abs(i - offset)

            best_fit = transform(x[minima][threshold][0])
            res[coefficient]['best fit'] = [(best_fit, y[transform(x).argsort()][minima][threshold][0])]

            y = y[transform(x)[x < offset].argsort()]
            x = np.array(sorted(transform(x)[x < offset]))

            res[coefficient]['one sigma'] = [line.interval(x, y, 1.0, best_fit)] if line.interval(x, y, 1.0, best_fit) else []
            res[coefficient]['two sigma'] = [line.interval(x, y, 3.84, best_fit)] if line.interval(x, y, 3.84, best_fit) else []

            sign = '+' if offset < 0 else '-'
            template = r'$|{}{}|$' if conversion_factor == 1. else r'$|{}/\Lambda^2{}|$'
            if round(offset, 1) != 0:
                res[coefficient]['label'] = template.format(
                    label[coefficient].replace('$', ''),
                    r' {} {:03.1f}{}'.format(sign, np.abs(offset), '\ \mathrm{TeV}^{-2}' if conversion_factor != 1. else '')
                )
            else:
                res[coefficient]['label'] = template.format(label[coefficient].replace('$', ''), '')
            res[coefficient]['transformed'] = True
        else:
            for xbf, ybf in zip(x[minima][threshold], y[minima][threshold]):
                res[coefficient]['best fit'].append((xbf, ybf))
            res[coefficient]['one sigma'] = []
            res[coefficient]['two sigma'] = []
            for xbf, ybf in zip(x[minima], y[minima]):
                if line.interval(x, y, 1.0, xbf) and line.interval(x, y, 1.0, xbf) not in res[coefficient]['one sigma']:
                    res[coefficient]['one sigma'].append(line.interval(x, y, 1.0, xbf))
                if line.interval(x, y, 3.84, xbf) and line.interval(x, y, 3.84, xbf) not in res[coefficient]['two sigma']:
                    res[coefficient]['two sigma'].append(line.interval(x, y, 3.84, xbf))
            template = r'${}$' if conversion_factor == 1. else r'${}/\Lambda^2$'
            res[coefficient]['label'] = template.format(label[coefficient].replace('$', ''))

        res[coefficient]['x'] = x
        res[coefficient]['y'] = y

    table = []
    for coefficient, info in res.items():
        table.append([
            info['label'],
            ', '.join(['{:.1f}'.format(round(x[0], 2) + 0) for x in info['best fit']]),
            ' and '.join(['[{:.1f}, {:.1f}]'.format(i, j) for i, j in info['one sigma']]),
            ' and '.join(['[{:.1f}, {:.1f}]'.format(i, j) for i, j in info['two sigma']])
        ])
    headers = ['Wilson coefficient', 'best fit', '$1\sigma$ CL', '$2\sigma$  CL']
    tag = '{}{}'.format(('_transformed' if transform else ''), ('_dimensionless' if dimensionless else ''))
    with open(os.path.join(config['outdir'], 'best_fit{}.txt'.format(tag)), 'w') as f:
        f.write(tabulate.tabulate(table, headers=headers))
    with open(os.path.join(config['outdir'], 'best_fit{}.tex'.format(tag)), 'w') as f:
        f.write(tabulate.tabulate(table, headers=headers, tablefmt='latex_raw'))

    np.save('nll{}{}.npy'.format('_transformed' if transform else '', '_dimensionless' if dimensionless else ''), res)

    return res
