"""Run the paper experiments and compare scores with the archived results."""

import argparse
import hashlib
import json
import os
import platform

import numpy as np
import pandas as pd
import sklearn
from threadpoolctl import threadpool_limits

from data import ROOT, save


def check(out):
    reference = json.loads((ROOT / 'reference.json').read_text(encoding='utf-8'))
    checked = 0
    for filename, section, keys, values in [
        ('main.json', 'metrics', ['model', 'subset', 'seed', 'source'], ['accuracy', 'macro_f1']),
        ('main.json', 'lookup', ['model', 'subset', 'source'], ['accuracy', 'macro_f1']),
        ('main.json', 'selection', ['step'], ['macro_f1']),
        ('dependence.json', 'dependence', ['subset', 'source'], ['accuracy', 'macro_f1', 'vectors', 'ceiling']),
        ('laboratory.json', 'laboratory', ['protocol', 'method'], ['balanced_accuracy', 'overlap']),
        ('laboratory.json', 'lab_predictions', ['protocol', 'fold', 'method', 'profile'], ['prediction']),
    ]:
        path = out / filename
        if not path.exists():
            raise FileNotFoundError(f'{path.name} is missing; run all experiments first.')
        result = json.loads(path.read_text(encoding='utf-8'))
        if section == 'dependence':
            actual = result
        elif section == 'lab_predictions':
            actual = result['predictions']
        elif section == 'laboratory':
            actual = result['metrics']
        elif section == 'selection':
            actual = [{'step': i + 1, 'macro_f1': r['validation']['combined']['macro_f1']}
                      for i, r in enumerate(result['selection'])]
            if result['compact'] != ['F02', 'F13']:
                raise AssertionError('Compact feature selection changed.')
        else:
            actual = result[section]
        index = {tuple(row[k] for k in keys): row for row in actual}
        expected = reference[section]
        expected_keys = {tuple(row[k] for k in keys) for row in expected}
        if set(index) != expected_keys or len(index) != len(actual):
            raise AssertionError(f'Different result rows: {section}')
        for row in expected:
            found = index[tuple(row[k] for k in keys)]
            for key in values:
                if isinstance(row[key], str):
                    if found[key] != row[key]:
                        raise AssertionError(f'{section}: {key} differs for {row}')
                else:
                    np.testing.assert_allclose(found[key], row[key], rtol=0, atol=1e-12,
                                               err_msg=f'{section}: {key}, {row}')
                checked += 1
    save(out / 'verification.json', {'comparisons': checked, 'scores_match': True,
                                   'costs_compared': False})
    print(f'PASS: {checked} score, prediction and representation comparisons. Costs are hardware-dependent.')


def plot(out):
    os.environ.setdefault('MPLCONFIGDIR', str(out / '.matplotlib'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    rows = json.loads((out / 'dependence.json').read_text(encoding='utf-8'))
    rows = [r for r in rows if r['stage'] >= 0 and r['source'] == 'combined']
    rows.sort(key=lambda r: r['stage'])
    fig, ax = plt.subplots(figsize=(6, 3.4))
    ax.plot([r['groups'] for r in rows], [100 * r['accuracy'] for r in rows], 'o-', label='RF100')
    ax.plot([r['groups'] for r in rows], [100 * r['ceiling'] for r in rows], '--',
            label='Vector-based accuracy ceiling')
    ax.invert_xaxis()
    ax.set(xlabel='Remaining feature groups', ylabel='Accuracy (%)', ylim=(0, 105))
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / 'progressive_removal.png', dpi=200)
    fig.savefig(out / 'progressive_removal.pdf')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('experiment', nargs='?', default='all',
                        choices=['all', 'main', 'dependence', 'laboratory', 'check'])
    args = parser.parse_args()
    out = ROOT / 'results'
    out.mkdir(exist_ok=True)
    files = ['data/main.csv.gz', 'data/laboratory.csv.gz']
    hashes = {f: hashlib.sha256((ROOT / f).read_bytes()).hexdigest() for f in files}
    expected = json.loads((ROOT / 'data/provenance.json').read_text(encoding='utf-8'))['files']
    if hashes != expected:
        raise ValueError('Input files differ from the saved paper data.')
    if args.experiment == 'check':
        check(out)
        return
    save(out / 'environment.json', {'python': platform.python_version(), 'platform': platform.platform(),
                                   'numpy': np.__version__, 'pandas': pd.__version__,
                                   'scikit-learn': sklearn.__version__,
                                   'inputs': hashes})
    with threadpool_limits(limits=1):
        if args.experiment in ['all', 'main']:
            from main import run
            run(out)
        if args.experiment in ['all', 'dependence']:
            from dependence import run
            run(out)
            plot(out)
        if args.experiment in ['all', 'laboratory']:
            from laboratory import run
            run(out)
    if args.experiment == 'all':
        check(out)


if __name__ == '__main__':
    main()
