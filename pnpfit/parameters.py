import numpy as np

nlo = {
    'ttW': 0.601,
    'ttZ': 0.839,
    'ttH': 0.496
}

names = {
    'Fake': 'fake',
    'WZ': 'WZ',
    'ZZ': 'ZZ',
    'charge': 'charge',
    'fake': 'fake',
    'rare': 'rare',
    'ttH': 'ttH',
    'ttW': 'ttW',
    'ttX': 'ttX',
    'ttZ': 'ttZ',
    'ttother': 'ttX'
}

label = {
    'sigma ttW': r'$\sigma_{\mathrm{t\bar{t}W}}$ $\mathrm{[pb]}$',
    'sigma ttZ': r'$\sigma_{\mathrm{t\bar{t}Z}}$ $\mathrm{[pb]}$',
    'ttW': r'$\mathrm{t\bar{t}W}$',
    'ttZ': r'$\mathrm{t\bar{t}Z}$',
    'ttH': r'$\mathrm{t\bar{t}H}$',
    'ttV': r'$\mathrm{t\bar{t}V}$',
    'tt': r'$\mathrm{t\bar{t}}$',
    'c2B': r'$\bar{c}_{2B}$',
    'c2G': r'$\bar{c}_{2G}$',
    'c2W': r'$\bar{c}_{2W}$',
    'c3G': r'$\bar{c}_{3G}$',
    'c3W': r'$\bar{c}_{3W}$',
    'c6': r'$\bar{c}_{6}$',
    'cA': r'$\bar{c}_{A}$',
    'cB': r'$\bar{c}_{B}$',
    'cG': r'$\bar{c}_{G}$',
    'cH': r'$\bar{c}_{H}$',
    'cHB': r'$\bar{c}_{HB}$',
    'cHL': r'$\bar{c}_{HL}$',
    'cHQ': r'$\bar{c}_{HQ}$',
    'cHW': r'$\bar{c}_{HW}$',
    'cHd': r'$\bar{c}_{Hd}$',
    'cHe': r'$\bar{c}_{He}$',
    'cHu': r'$\bar{c}_{Hu}$',
    'cHud': r'$\bar{c}_{Hud}$',
    'cT': r'$\bar{c}_{T}$',
    'cWW': r'$\bar{c}_{WW}$',
    'cd': r'$\bar{c}_{d}$',
    'cdB': r'$\bar{c}_{dB}$',
    'cdG': r'$\bar{c}_{dG}$',
    'cdW': r'$\bar{c}_{dW}$',
    'cl': r'$\bar{c}_{l}$',
    'clB': r'$\bar{c}_{lB}$',
    'clW': r'$\bar{c}_{lW}$',
    'cpHL': r"$\bar{c}'_{HL}$",
    'cpHQ': r"$\bar{c}'_{HQ}$",
    'cu': r'$\bar{c}_{u}$',
    'cuB': r'$\bar{c}_{uB}$',
    'cuG': r'$\bar{c}_{uG}$',
    'cuW': r'$\bar{c}_{uW}$',
    'tc3G': r'$\widetilde{c}_{3G}$',
    'tc3W': r'$\widetilde{c}_{3W}$',
    'tcA': r'$\widetilde{c}_{A}$',
    'tcG': r'$\widetilde{c}_{G}$',
    'tcHB': r'$\widetilde{c}_{HB}$',
    'tcHW': r'$\widetilde{c}_{HW}$',
}

# see http://feynrules.irmp.ucl.ac.be/attachment/wiki/HEL/HEL.fr
vev = 0.246  # TeV
mass_w = 0.080385  # TeV
mass_higgs = 0.125  # TeV
higgs_quartic_coupling = np.power(mass_higgs, 2) / (2 * np.power(vev, 2))  # see http://sns.ias.edu/~pgl/SMB/Higgs_update.pdf equation 7.196
ymt = 0.172  # TeV
yu = np.sqrt(2) * ymt / vev  # up-type Yukawa couplings (the first two generations have been removed from the model)
ymb = 0.0047  # TeV
yb = np.sqrt(2) * ymb / vev  # up-type Yukawa couplings (the first two generations have been removed from the model)
aeW = 1. / 127.9  # electroweak coupling contant
ee = np.sqrt(4 * np.pi * aeW)  # electric coupling constant
sw = ee * vev / (2 * mass_w)  # sine of the Weinberg angle
gw = ee / sw  # weak coupling constant at the Z pole
g1 = ee / np.sqrt(1 - sw * sw)  # U(1)Y coupling constant at the Z pole
aS = 0.1184
gs = np.sqrt(4 * np.pi * aS)  # strong coupling constant at the Z pole


# see https://arxiv.org/pdf/1310.5150v2.pdf equations 2.4 - 2.14
# see http://feynrules.irmp.ucl.ac.be/attachment/wiki/HEL/HEL.fr
conversion = {
    'cH': 1. / (2 * vev * vev),
    'cT': 1. / (2 * vev * vev),
    'c6': higgs_quartic_coupling / (vev * vev),
    'cu': yu / (vev * vev),
    'cWW': gw / (mass_w * mass_w),
    'cB': g1 / (2 * mass_w * mass_w),
    'cHW': 2 * gw / (mass_w * mass_w),
    'cHB': g1 / (mass_w * mass_w),
    'cA': g1 * g1 / (mass_w * mass_w),
    'cG': gs * gs / (mass_w * mass_w),
    'cHQ': 1. / (vev * vev),
    'cpHQ': 4. / (vev * vev),
    'cHu': 1. / (2 * vev * vev),
    'cHd': 1. / (2 * vev * vev),
    'cHud': 1. / (vev * vev),
    'cHL': 1. / (vev * vev),
    'cHe': 1. / (2 * vev * vev),
    'cuB': g1 * yu / (2 * mass_w * mass_w),
    'cuW': gw * yu / (mass_w * mass_w),
    'cuG': gs * yu / (mass_w * mass_w),
    'c3W': gw * gw * gw / (mass_w * mass_w),
    'c3G': gs * gs * gs / (mass_w * mass_w),
    'c2W': 1. / (mass_w * mass_w),
    'c2B': 1. / (mass_w * mass_w),
    'c2G': 1. / (mass_w * mass_w),
    'tcHW': gw / (mass_w * mass_w),
    'tcHB': g1 / (2 * mass_w * mass_w),
    'tcA': g1 * g1 / (2 * mass_w * mass_w),
    'tc3W': gw * gw * gw / (2 * mass_w * mass_w),
    'tc3G': gs * gs * gs / (mass_w * mass_w),
    'cpHL': 4. / (vev * vev),
    'tcG': gs * gs / (2. * mass_w * mass_w)
}
