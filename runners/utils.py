import os


def get_doc(doc, baseline, augment, positive, feature_size, arch):
    if baseline:
        doc += 'Baseline'
    if augment:
        doc += 'a'
    if positive:
        doc += 'p'
    if feature_size > 0:
        doc += str(feature_size)
    doc += arch.lower()
    return doc


def check_mcc(dset, baseline=False, augment=False, positive=False, feature_size=0, arch='ConvMLP'):
    doc = get_doc('representation', baseline, augment, positive, feature_size, arch)
    path = os.path.join('run', 'output', dset, doc)
    files = os.listdir(path)
    acc = []
    for i in range(19):
        for j in range(i + 1, 20):
            if 'mcc_strong_{}_{}.p'.format(i, j) not in files or 'mcc_weak_{}_{}.p'.format(i, j) not in files:
                acc.append((i, j))
    if len(acc) > 0:
        print('\n[MCC] d: {}; b: {}; a: {}; p: {}; f: {}'.format(dset, baseline, augment, positive, feature_size))
        print('Missing MCCs: {}'.format(acc))


def check_rep(dset, baseline=False, augment=False, positive=False, feature_size=0, arch='ConvMLP'):
    doc = get_doc('representation', baseline, augment, positive, feature_size, arch)
    path = os.path.join('run', 'checkpoints', dset, doc)
    files = os.listdir(path)
    acc1 = []
    acc2 = []
    for i in range(20):
        if 'seed{}'.format(i) not in files:
            acc1.append(i)
    for f in files:
        if 'test_representations.p' not in os.listdir(os.path.join(path, f)):
            acc2.append(f)
    if len(acc1) > 0 or len(acc2) > 0:
        print('\n[REP] d: {}; b: {}; a: {}; p: {}; f: {}'.format(dset, baseline, augment, positive, feature_size))
        if len(acc1) > 0:
            print('Representations missing for seeds {}'.format(acc1))
        if len(acc2) > 0:
            print('Representations coorupted for {}'.format(acc2))


def check_transfer(dset, baseline=False, augment=False, positive=False, feature_size=0, arch='ConvMLP'):
    doc = get_doc('transfer', baseline, augment, positive, feature_size, arch)
    path = os.path.join('run', 'output', dset, doc)
    files = os.listdir(path)
    acc = []
    for i in [0, 500, 1000, 2000, 3000, 4000, 5000, 6000]:
        for j in range(5):
            if 'all_epochs_SIZE{}_SEED{}.p'.format(i, j) not in files:
                acc.append((i, j))
    if len(acc) > 0:
        print('\n[TRANSFER] d: {}; b: {}; a: {}; p: {}; f: {}'.format(dset, baseline, augment, positive, feature_size))
        print('Missing transfer output for size/seed: {}'.format(acc))


def check_all(dset, baseline=False, augment=False, positive=False, feature_size=0, arch='ConvMLP'):
    check_transfer(dset, baseline, augment, positive, feature_size, arch)
    check_rep(dset, baseline, augment, positive, feature_size, arch)
    check_mcc(dset, baseline, augment, positive, feature_size, arch)
