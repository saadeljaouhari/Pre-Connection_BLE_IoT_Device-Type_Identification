"""RF100 progressive removal and feature combinations (Figure 3 and Table IV)."""

from collections import Counter, defaultdict

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from threadpoolctl import threadpool_limits

from data import GROUPS, columns, load_main, save, source_scores

ORDER = ['F01', 'F06', 'F13', 'F10', 'F07', 'F05', 'F19',
         'F18', 'F02', 'F03', 'F04', 'F17', 'F11']
COMBINATIONS = {
    'B': ['F04', 'F11', 'F14', 'F17'],
    'B+F03': ['F03', 'F04', 'F11', 'F14', 'F17'],
    'B+F18': ['F04', 'F11', 'F14', 'F17', 'F18'],
    'B+F03+F18': ['F03', 'F04', 'F11', 'F14', 'F17', 'F18'],
    'F03': ['F03'], 'F18': ['F18'], 'F03+F18': ['F03', 'F18'],
}


def run(out):
    frame, features = load_main()
    # The diagnostic scripts read the historical float32 export as float64.
    features = features.astype(np.float64)
    train, test = frame['split'].eq('train'), frame['split'].eq('test')
    encoder = LabelEncoder().fit(frame.loc[train, 'Device_Type'])
    y_train, truth = encoder.transform(frame.loc[train, 'Device_Type']), encoder.transform(frame.loc[test, 'Device_Type'])
    sources = frame.loc[test, 'source'].to_numpy()
    remaining, experiments = GROUPS.copy(), []
    for stage in range(len(ORDER) + 1):
        if stage:
            remaining.remove(ORDER[stage - 1])
        experiments.append((f'{len(remaining)}F', remaining.copy(), stage))
    experiments += [(name, groups, -1) for name, groups in COMBINATIONS.items()]
    results = []
    for name, groups, stage in experiments:
        cols = columns(features, groups)
        model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42, n_jobs=-1)
        model.fit(features.loc[train, cols].to_numpy(), y_train)
        matrix = features.loc[test, cols].to_numpy()
        prediction = model.predict(matrix)
        for source, metrics in source_scores(truth, prediction, sources).items():
            mask = np.ones(len(truth), dtype=bool) if source == 'combined' else sources == source
            buckets = defaultdict(Counter)
            for vector, label in zip(map(tuple, matrix[mask]), truth[mask]):
                buckets[vector][int(label)] += 1
            results.append({'subset': name, 'stage': stage, 'groups': len(groups), 'source': source,
                            'features': groups, 'vectors': len(buckets),
                            'ceiling': sum(max(v.values()) for v in buckets.values()) / int(mask.sum()), **metrics})
        print(f'RF100: {name}', flush=True)
    save(out / 'dependence.json', results)


if __name__ == '__main__':
    from data import ROOT
    with threadpool_limits(limits=1):
        run(ROOT / 'results')
