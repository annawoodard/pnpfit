import numpy as np
import logging

from HiggsAnalysis.CombinedLimit.PhysicsModel import PhysicsModel

from NPFitProduction.NPFitProduction.cross_sections import CrossSectionScan
from NPFitProduction.NPFitProduction.utils import sorted_combos

logger = logging.getLogger(__name__)

class EFTScaling(PhysicsModel):
    """Apply process scaling due to EFT operators.

    This class takes a `CrossSectionScan`,  performs a fit to describe how processes are
    scaled as a function of an EFT operator's Wilson coefficient and adds it to
    the workspace.

    """

    def setPhysicsOptions(self, options):
        self.pois = []
        self.processes = []
        for option, value in [x.split('=') for x in options]:
            if option == 'poi':
                self.pois.append(value)
            if option == 'process':  # processes which will be scaled
                self.processes.append(value)
            if option == 'scan':
                self.scan = CrossSectionScan(value)

    def setup(self):
        # self.scan.fit(dimensions=[2])
        self.scan.fit()
        for process in self.processes:
            if process not in self.scan.fit_constants:
                raise RuntimeError('no fit provided for process {}'.format(process))
            self.modelBuilder.out.var(process)
            name = 'r_{0}'.format(process)

            pairs = sorted_combos(range(0, len(self.pois)), 2)
            fit_constants = self.scan.construct(process, tuple(self.pois))

            constant = ['1.0']
            linear = self.pois
            quad = ['{p} * {p}'.format(p=p) for p in self.pois]
            mixed = ['{p0} * {p1}'.format(p0=self.pois[p0], p1=self.pois[p1]) for p0, p1 in pairs]
            terms = ['({s:0.11f} * {c})'.format(s=s, c=c) for s, c in zip(fit_constants, constant + linear + quad + mixed)]
            model = 'expr::{name}("{terms}", {pois})'.format(name=name, terms=' + '.join(terms), pois=', '.join(self.pois))

            scale = self.modelBuilder.factory_(model.replace(' ', ''))
            self.modelBuilder.out._import(scale)

    def doParametersOfInterest(self):
        # convergence of the loop expansion requires c < (4 * pi)^2
        # see section 7 https://arxiv.org/pdf/1205.4231.pdf
        cutoff = (4 * np.pi) ** 2
        for poi in self.pois:
            self.modelBuilder.doVar('{poi}[0, -{cutoff}, {cutoff}]'.format(poi=poi, cutoff=cutoff))

        self.modelBuilder.doSet('POI', ','.join(self.pois))

        self.setup()

    def getYieldScale(self, bin, process):
        if process not in self.processes:
            return 1
        else:
            name = 'r_{0}'.format(process)

            return name


eft = EFTScaling()
