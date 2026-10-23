"""Main classifiers, feature selection, baselines and costs (Tables III and V)."""

import copy
import time
from collections import Counter, defaultdict

import joblib
import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import log_loss
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.class_weight import compute_sample_weight
from threadpoolctl import threadpool_limits

from data import GROUPS, PRIMARY, SETS, columns, load_main, save, source_scores
from data import scores


def fit_model(kind, train, validation, y_train, y_val, n_classes, seed):
    if kind == 'RF':
        model = RandomForestClassifier(n_estimators=200, max_features='sqrt',
                                       random_state=seed, n_jobs=2)
        model.fit(train, y_train, sample_weight=compute_sample_weight('balanced', y_train))
        model.n_jobs = 1
    elif kind == 'MLP':
        scaler = StandardScaler().fit(train)
        train_scaled, val_scaled = scaler.transform(train), scaler.transform(validation)
        net = MLPClassifier(hidden_layer_sizes=(128, 128), activation='relu', solver='adam',
                            alpha=1e-4, batch_size=256, learning_rate_init=1e-3,
                            shuffle=True, random_state=seed, early_stopping=False, max_iter=1)
        best, best_loss, waiting = None, np.inf, 0
        weights = compute_sample_weight('balanced', y_train)
        for _ in range(200):
            net.partial_fit(train_scaled, y_train, classes=np.arange(n_classes), sample_weight=weights)
            loss = log_loss(y_val, net.predict_proba(val_scaled), labels=np.arange(n_classes))
            if loss < best_loss - 1e-5:
                best, best_loss, waiting = copy.deepcopy(net), loss, 0
            else:
                waiting += 1
            if waiting >= 20:
                break
        if best is None:
            raise RuntimeError('No finite MLP validation checkpoint.')
        model = Pipeline([('scaler', scaler), ('classifier', best)])
    elif kind == 'DT':
        candidates = []
        for depth in [2, 4, 6, 8, 12, None]:
            tree = DecisionTreeClassifier(max_depth=depth, class_weight='balanced', random_state=seed)
            tree.fit(train, y_train)
            score = scores(y_val, tree.predict(validation))['macro_f1']
            candidates.append((-score, tree.tree_.node_count, tree))
        model = sorted(candidates, key=lambda item: item[:2])[0][2]
    else:
        model = DummyClassifier(strategy='most_frequent').fit(train, y_train)
    return model


def run(out):
    frame, features = load_main()
    indices = {s: np.flatnonzero(frame['split'].eq(s)) for s in ['train', 'val', 'test']}
    labels = LabelEncoder().fit(frame.iloc[indices['train']]['Device_Type'])
    y = labels.transform(frame['Device_Type'])
    sources = frame['source'].to_numpy()

    def train(kind, groups, seed):
        # Preserve the original group order, including during forward selection.
        cols = columns(features, [g for g in GROUPS if g in groups])
        start = time.perf_counter()
        model = fit_model(kind, features.iloc[indices['train']][cols].to_numpy(),
                          features.iloc[indices['val']][cols].to_numpy(),
                          y[indices['train']], y[indices['val']], len(labels.classes_), seed)
        duration = time.perf_counter() - start
        pred = model.predict(features.iloc[indices['val']][cols].to_numpy())
        validation = source_scores(y[indices['val']], pred, sources[indices['val']])
        return model, cols, duration, validation

    selection_start = time.perf_counter()
    _, _, _, reference = train('RF', PRIMARY, 42)
    selected, remaining, selection = [], PRIMARY.copy(), []
    while remaining:
        candidates = []
        for group in remaining:
            _, cols, _, metrics = train('RF', selected + [group], 42)
            candidates.append((group, cols, metrics))
        candidates.sort(key=lambda item: (-item[2]['combined']['macro_f1'],
                                         -item[2]['combined']['accuracy'], len(item[1]), item[0]))
        group, _, metrics = candidates[0]
        selected.append(group)
        remaining.remove(group)
        selection.append({'added_feature': group, 'features': selected.copy(), 'validation': metrics})
        if all(metrics[s]['macro_f1'] >= reference[s]['macro_f1'] - .005 for s in reference):
            break
    selection_seconds = time.perf_counter() - selection_start
    compact = [g for g in GROUPS if g in selected]
    sets = {**SETS, 'compact_no_headers': compact}
    sets.update({'single_' + g: [g] for g in GROUPS})
    sets.update({'primary_minus_' + g: [h for h in PRIMARY if h != g] for g in PRIMARY})

    test = indices['test']
    audits, lookup = [], []
    for name, groups in sets.items():
        keys = list(map(tuple, features[columns(features, groups)].to_numpy()))
        counts = defaultdict(Counter)
        for i in indices['train']:
            counts[keys[i]][int(y[i])] += 1
        pred = np.array([min(counts[keys[i]], key=lambda c: (-counts[keys[i]][c], c))
                         if keys[i] in counts else -1 for i in test])
        for source, metrics in source_scores(y[test], pred, sources[test]).items():
            mask = np.ones(len(test), dtype=bool) if source == 'combined' else sources[test] == source
            buckets = defaultdict(Counter)
            for i in test[mask]:
                buckets[keys[i]][int(y[i])] += 1
            audits.append({'subset': name, 'source': source,
                           'distinct_vectors': len(buckets),
                           'overlap': float(np.mean([keys[i] in counts for i in test[mask]])),
                           'ceiling': sum(max(v.values()) for v in buckets.values()) / int(mask.sum())})
            lookup.append({'model': 'Lookup', 'subset': name, 'source': source, **metrics})

    jobs = [('RF', name, seed) for name in sets for seed in [42, 43, 44]]
    jobs += [(kind, name, seed) for kind, name in [
        ('MLP', 'full'), ('MLP', 'primary_no_headers'),
        ('DT', 'primary_no_headers'), ('Majority', 'primary_no_headers')]
        for seed in [42, 43, 44]]
    metrics, costs = [], []
    (out / 'models').mkdir(parents=True, exist_ok=True)
    for number, (kind, name, seed) in enumerate(jobs, 1):
        print(f'Main {number}/{len(jobs)}: {kind}, {name}, seed {seed}', flush=True)
        model, cols, duration, _ = train(kind, sets[name], seed)
        matrix = features.iloc[test][cols].to_numpy()
        pred = model.predict(matrix)
        batch = matrix[:1000]
        model.predict(batch)
        timings = []
        for _ in range(7):
            start = time.perf_counter()
            model.predict(batch)
            timings.append(time.perf_counter() - start)
        if (kind, name) in [('RF', 'full'), ('RF', 'primary_no_headers'), ('RF', 'compact_no_headers'),
                            ('MLP', 'primary_no_headers'), ('DT', 'primary_no_headers')]:
            path = out / 'models' / f'{kind}_{name}_{seed}.joblib'
            joblib.dump({'model': model, 'columns': cols, 'classes': labels.classes_.tolist()}, path, compress=3)
            costs.append({'model': kind, 'subset': name, 'seed': seed,
                          'fit_seconds': duration, 'prediction_us_per_row': float(1e6 * np.median(timings) / len(batch)),
                          'model_bytes': path.stat().st_size})
        for source, values in source_scores(y[test], pred, sources[test]).items():
            metrics.append({'model': kind, 'subset': name, 'seed': seed, 'source': source, **values})
    save(out / 'main.json', {'metrics': metrics, 'lookup': lookup, 'representations': audits,
                            'compact': compact, 'selection': selection, 'selection_seconds': selection_seconds,
                            'costs': costs})


if __name__ == '__main__':
    from data import ROOT
    with threadpool_limits(limits=1):
        run(ROOT / 'results')
