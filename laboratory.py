"""Four-category tests with complete laboratory profiles withheld (Table VI)."""

from collections import Counter

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier

from data import ROOT, save

FIELDS = ['adv_payload_length', 'adv_entry_length', 'adv_entry_type', 'scan_payload_length',
          'scan_name_entry_length', 'scan_name_entry_type',
          'scan_manufacturer_entry_length', 'scan_manufacturer_entry_type']
CATEGORICAL = ['adv_entry_type', 'scan_name_entry_type', 'scan_manufacturer_entry_type']
SEED = 20261009


def matrix(frame):
    result = frame[FIELDS].astype(float).copy()
    # All descriptors are present. Keep these constant indicators to preserve
    # the original tree feature indices and RF feature sampling.
    for field in FIELDS:
        result[field + '__available'] = 1
    return result


def protocols(frame):
    lab = frame.loc[frame['source'].eq('laboratory')]
    synthetic = frame.loc[frame['source'].eq('historical_synthetic')]
    train, test = lab.loc[lab['split'].eq('train')], lab.loc[lab['split'].eq('test')]
    yield 'synthetic_to_lab', 'all_lab', synthetic, lab
    yield 'lab_row_split_diagnostic', 'saved_split', train, test
    yield 'mixed_row_split_diagnostic', 'saved_split', pd.concat([synthetic, train]), test
    for profile in lab['profile'].unique():
        yield ('mixed_lab_profile_holdout', 'hold_' + profile,
               pd.concat([synthetic, lab.loc[lab['profile'].ne(profile)]]),
               lab.loc[lab['profile'].eq(profile)])


def run(out):
    frame = pd.read_csv(ROOT / 'data/laboratory.csv.gz', keep_default_na=False)
    if len(frame) != 5272 or frame['profile'].nunique() != 8:
        raise ValueError('Unexpected laboratory records.')
    predictions = []
    for protocol, fold, train, test in protocols(frame):
        if set(train['sample_id']) & set(test['sample_id']):
            raise ValueError('Training and testing records overlap.')
        train = train.drop_duplicates(FIELDS + ['Device_Type'])
        unique_test = test.drop_duplicates('profile')
        counts = test['profile'].value_counts()
        x_train, x_test = matrix(train), matrix(unique_test)
        transform = ColumnTransformer([
            ('categorical', OneHotEncoder(handle_unknown='ignore', sparse_output=False), CATEGORICAL),
            ('numeric_and_masks', 'passthrough', [c for c in x_train if c not in CATEGORICAL])])
        encoded_train = transform.fit_transform(x_train)
        encoded_test = transform.transform(x_test)
        known = dict(zip(map(tuple, train[FIELDS].to_numpy()), train['Device_Type']))
        frequency = Counter(train['Device_Type'])
        majority = min(frequency, key=lambda label: (-frequency[label], label))
        methods = {
            'ExactLookup': [known.get(tuple(row), 'ABSTAIN') for row in unique_test[FIELDS].to_numpy()],
            'MajorityProfile': [majority] * len(unique_test),
        }
        for name, model in [('DecisionTree', DecisionTreeClassifier(random_state=SEED)),
                            ('RandomForest', RandomForestClassifier(n_estimators=100, random_state=SEED, n_jobs=1))]:
            model.fit(encoded_train, train['Device_Type'])
            methods[name + '_forced'] = model.predict(encoded_test).tolist()
        for method, values in methods.items():
            for (_, row), prediction in zip(unique_test.iterrows(), values):
                predictions.append({'protocol': protocol, 'fold': fold, 'method': method,
                                    'profile': row['profile'], 'target': row['Device_Type'],
                                    'prediction': prediction, 'records': int(counts[row['profile']]),
                                    'vector_seen': tuple(row[FIELDS]) in known})
    metrics = []
    for protocol in dict.fromkeys(p['protocol'] for p in predictions):
        for method in ['DecisionTree_forced', 'RandomForest_forced', 'ExactLookup', 'MajorityProfile']:
            rows = [p for p in predictions if p['protocol'] == protocol and p['method'] == method]
            if len(rows) != 4 or len({r['target'] for r in rows}) != 4:
                raise ValueError('Each protocol must evaluate all four categories.')
            metrics.append({'protocol': protocol, 'method': method,
                            'balanced_accuracy': float(np.mean([r['target'] == r['prediction'] for r in rows])),
                            'overlap': float(np.mean([r['vector_seen'] for r in rows]))})
    save(out / 'laboratory.json', {'metrics': metrics, 'predictions': predictions})
    print('Laboratory: four protocols complete.', flush=True)


if __name__ == '__main__':
    run(ROOT / 'results')
