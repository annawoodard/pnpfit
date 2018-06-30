import os


def prepare_cards(args, config, cardify):
    for analysis, path in config['cards'].items():
        if os.path.isdir(path):
            subprocess.call('combineCards.py {} > {}'.format(os.path.join(path, '*.txt'), cardify(analysis)), shell=True)
        elif os.path.isfile(path):
            shutil.copy(path, cardify(analysis))

    with open(cardify('4l'), 'r') as f:
        card = f.read()
    with open(cardify('4l'), 'w') as f:
        # TODO fix this
        f.write(card[:card.find('nuisance parameters') + 19])
        f.write('''
----------------------------------------------------------------------------------------------------------------------------------
shapes *      ch1  FAKE
shapes *      ch2  FAKE''')
        f.write(card[card.find('nuisance parameters') + 19:])

    subprocess.call('combineCards.py {} {} > {}'.format(cardify('3l'), cardify('4l'), cardify('ttZ')), shell=True)
    subprocess.call('cp {} {}'.format(cardify('2l'), cardify('ttW')), shell=True)
    subprocess.call('combineCards.py {} {} > {}'.format(cardify('ttZ'), cardify('ttW'), cardify('ttV_np')), shell=True)

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