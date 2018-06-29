import os
import tabulate as tb
import matplotlib.pyplot as plt
import numpy as np
from NPFit.NPFit.makeflow import multidim_np
from NPFit.NPFit.parameters import conversion, label
from root_numpy import root2array
from NPFitProduction.NPFitProduction.utils import sorted_combos
from scipy.stats import chi2

def round(num, sig_figs):
    return str(float('{0:.{1}e}'.format(num, sig_figs - 1)))

class CLIntervals(object):
    def __init__(self, dimension, dimensionless=False, levels=[0.68, 0.95], freeze=False):
        self.dimension = dimension
        self.dimensionless = dimensionless
        self.levels = levels
        self.base = 'cl_intervals_{}d_{}{}{}'.format(
            dimension,
            '_'.join(['cl{}'.format(x) for x in levels]),
            '_dimensionless' if dimensionless else '',
            '_non_pois_fixed_to_zero' if freeze else ''
        )
        self.freeze = freeze

    def specify(self, config, spec, index):
        inputs = multidim_np(config, spec, self.dimension, cl=self.levels, freeze=self.freeze)
        outputs = [os.path.join(config['outdir'], self.base + extension) for extension in ['.tex', '.txt']]
        spec.add(inputs, outputs, ['run', 'tabulate', '-i', str(index), config['fn']])

    def write(self, config, args):
        sf = 2
        if self.dimension == 2:
            def contiguous(segments):
                for i in range(len(segments) - 1):
                    if max(segments[i]) < min(segments[i + 1]):
                        return False
                return True
            coefficients = sorted(config['coefficients'])
            indices = dict((i, c) for c, i in enumerate(coefficients))
            tables = dict((level, [[c] + ['-'] * len(coefficients) for c in coefficients]) for level in self.levels)
            for x, y in sorted_combos(config['coefficients'], 2):
                tag = '{}_{}{}'.format(x, y, '_frozen' if self.freeze else '')
                data = root2array(os.path.join(config['outdir'], 'scans/{}.total.root'.format(tag)))
                zi = 2 * data['deltaNLL']
                xi = data[x]
                yi = data[y]
                if not self.dimensionless:
                    xi *= conversion[x]
                    yi *= conversion[y]

                contour = plt.tricontour(
                    xi[zi != 0],
                    yi[zi != 0],
                    zi[zi != 0],
                    sorted(chi2.isf([1 - l for l in self.levels], 2))
                )
                for i, l in enumerate(self.levels):
                    res = {
                        x: [],
                        y: []
                    }
                    for path in contour.collections[i].get_paths():
                        polygons = path.to_polygons()
                        for p in polygons:
                            res[x].append((p[:, 0].min(), p[:, 0].max()))
                            res[y].append((p[:, 1].min(), p[:, 1].max()))
                    for key in res:
                        res[key].sort()
                        if contiguous(res[key]):
                            flattened = sum(res[key], ())
                            res[key] = [(min(flattened), max(flattened))]
                    intervals = ['[{}, {}]'.format(round(low, sf), round(high, sf)) for low, high in res[x]]
                    tables[l][indices[x]][indices[y] + 1] = ' and '.join(intervals)
                    intervals = ['[{}, {}]'.format(round(low, sf), round(high, sf)) for low, high in res[y]]
                    tables[l][indices[y]][indices[x] + 1] = ' and '.join(intervals)

            with open(os.path.join(config['outdir'], self.base + '.txt'), 'w') as f:
                headers = [""] + [x + ('' if self.dimensionless else '/Lambda^2') for x in coefficients]
                for level, table in tables.items():
                    f.write('\n\n{line} cl={level} {line}\n{text}'.format(
                        line='#' * 20,
                        level=level,
                        text=tb.tabulate(table, headers=headers))
                    )

            with open(os.path.join(config['outdir'], self.base + '.tex'), 'w') as f:
                headers = [""] + [x + ('' if self.dimensionless else '$/\Lambda^2$') for x in coefficients]
                for level, table in tables.items():
                    f.write('\n\n{line} cl={level} {line}\n{text}'.format(
                        line='#' * 20,
                        level=level,
                        text=tb.tabulate(table, headers=headers, tablefmt='latex_raw'))
                    )
        else:
            table = []
            for coefficients in sorted_combos(config['coefficients'], self.dimension):
                for index, coefficient in enumerate(sorted(coefficients)):
                    row = [coefficient]
                    for level in self.levels:
                        tag = '{}{}'.format('_'.join(coefficients), '_frozen' if self.freeze else '')
                        data = root2array(
                            os.path.join(config['outdir'], 'cl_intervals/{}-{}.root'.format(tag, level))
                        )
                        template = '[{}, {}]'
                        conversion_factor = 1. if self.dimensionless else conversion[coefficient]
                        low = data[coefficient][(index + 1) * 2 - 1] * conversion_factor
                        high = data[coefficient][(index + 1) * 2] * conversion_factor
                        row += [template.format(round(low, sf), round(high, sf))]
                    row += ['{}'.format(round(data[coefficient][0], sf + 1))]
                    table.append(row)

            headers = [''] + ['CL={}'.format(cl) for cl in self.levels] + ['best fit']
            with open(os.path.join(config['outdir'], self.base + '.txt'), 'w') as f:
                f.write('dimension: {} {}\n'.format(self.dimension, '' if self.dimensionless else '(all values in [TeV^(-2)])'))
                f.write(tb.tabulate(
                    [[x[0] + ('' if self.dimensionless else '/\Lambda^2')] + x[1:] for x in table],
                    headers,
                    tablefmt='plain') + '\n'
                )
            with open(os.path.join(config['outdir'], self.base + '.tex'), 'w') as f:
                table = [[label[x[0]] + ('' if self.dimensionless else '$/\Lambda^2$')] + x[1:] for x in table]
                f.write('dimension: {} {}\n'.format(self.dimension, '' if self.dimensionless else '(all values in [TeV^(-2)])'))
                f.write(tb.tabulate(table, headers, tablefmt='latex_raw') + '\n')

def tabulate(args, config):
    if args.index is not None:
        config['tables'][args.index].write(config, args)
    else:
        for p in config['tables']:
            p.write(config, args)
