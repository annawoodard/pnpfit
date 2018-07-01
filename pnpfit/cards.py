import glob
import os
import shutil
import re


def prepare_cards(config):
    def cardify(name):
        return os.path.join(config['outdir'], '{}.txt'.format(name))

    for analysis, path in config['cards'].items():
        if os.path.isfile(path):
            shutil.copy(path, cardify(analysis))

    with open(cardify('ttV_np'), 'r') as f:
        card = f.read()

    processes = re.compile(r'\nprocess.*')

    for index, process in enumerate(['ttW', 'ttZ']):
        names, numbers = processes.findall(card)
        for column in [i for i, name in enumerate(names.split()) if name == process]:
            number = numbers.split()[column]
            card = card.replace(numbers, numbers.replace(number, '{}'.format(index * -1)))

    jmax = re.search('jmax (\d*)', card).group(0)
    card = card.replace(jmax, 'jmax {}'.format(len(set(names.split()[1:])) - 1))

    with open(cardify('ttW-ttZ'), 'w') as f:
        f.write(card)

    systematics = {}
    for label, info in config['systematics'].items():
        systematics[label] = '\n{label}                  {dist}     '.format(label=label, dist=info['distribution'])

    def compose(kappa):
        if kappa['-'] == kappa['+']:
            return str(kappa['+'])
        else:
            return '{}/{}'.format(kappa['-'], kappa['+'])

    for name in names.split()[1:]:
        for label, info in config['systematics'].items():
            systematics[label] += '{:15s}'.format(compose(info['kappa'][name]) if name in info['kappa'] else '-')

    kmax = re.search('kmax (\d*)', card).group(0)
    card = card.replace(kmax, 'kmax {}'.format(int(re.search('kmax (\d*)', card).group(1)) + 4))

    for line in card.split('\n'):
        if line.startswith('ttX'):
            card = re.sub(line, '#' + line, card)

    with open(cardify('ttV_np'), 'w') as f:
        f.write(card[:card.find('\ntheo group')])
        for line in systematics.values():
            f.write(line)
