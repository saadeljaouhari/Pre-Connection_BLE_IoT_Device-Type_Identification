"""Read the saved records and encode the paper's retained features."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

ROOT = Path(__file__).resolve().parent
GROUPS = ['F01', 'F02', 'F03', 'F04', 'F05', 'F06', 'F07',
          'F10', 'F11', 'F13', 'F14', 'F17', 'F18', 'F19']
PRIMARY = [g for g in GROUPS if g not in ('F01', 'F06')]
SETS = {
    'full': GROUPS,
    'primary_no_headers': PRIMARY,
    'advertisement_no_header': ['F02', 'F03', 'F04', 'F05'],
    'scan_response_no_header': ['F07', 'F10', 'F11', 'F13', 'F14'],
    'channel_timing': ['F17', 'F18', 'F19'],
}
# Group, stored field, encoded columns, encoding.
FIELDS = [
    ('F01', 'ADV_IND_advertising_header', ['ADV_hdr_b0', 'ADV_hdr_b1'], 'hex'),
    ('F02', 'ADV_IND_length', ['ADV_len'], 'length'),
    ('F03', 'ADV_IND_entry.length_1', ['ADV_entry_len1'], 'length'),
    ('F04', 'ADV_IND_entry.type_1', ['ADV_type1'], 'byte'),
    ('F05', 'ADV_IND_entry.uuid_16', ['UUID16_b0', 'UUID16_b1'], 'hex'),
    ('F06', 'SCAN_RSP_advertising_header', ['SCN_hdr_b0', 'SCN_hdr_b1'], 'hex'),
    ('F07', 'SCAN_RSP_length', ['SCN_len'], 'length'),
    ('F10', 'SCAN_RSP_entry.length_2', ['SCN_len2'], 'length'),
    ('F11', 'SCAN_RSP_entry.type_2', ['SCN_type2'], 'byte'),
    ('F13', 'SCAN_RSP_entry.length_3', ['SCN_len3'], 'length'),
    ('F14', 'SCAN_RSP_entry.type_3', ['SCN_type3'], 'byte'),
    ('F17', 'Channel_sequence', ['channel_pos1', 'channel_pos2', 'channel_pos3'], 'channel'),
    ('F18', 'Event_interval', ['event_interval'], 'float'),
    ('F19', 'Advertising_delay', ['advertising_delay'], 'float'),
]


def encode(frame):
    result = {}
    for group, field, names, encoding in FIELDS:
        rows = []
        for value in frame[field]:
            if encoding == 'channel':
                channels = [int(c) for c in str(value).split('|')]
                if len(channels) != 3 or any(c not in (37, 38, 39) for c in channels):
                    raise ValueError(f'Invalid channel sequence: {value}')
                row = [(c - 37) / 2 for c in channels]
            elif encoding == 'float':
                row = [float(value)]
            elif encoding == 'length':
                number = float(value)
                if not 0 <= number <= 255:
                    raise ValueError(f'Invalid length: {value}')
                row = [number / 255]
            else:
                text = str(value).strip().lower()
                number = int(text, 16) if text.startswith('0x') else int(float(text))
                row = [b / 255 for b in number.to_bytes(len(names), 'big')]
            rows.append(row)
        values = np.asarray(rows, dtype=np.float32)
        for i, name in enumerate(names):
            result[group + '_' + name] = values[:, i]
    encoded = pd.DataFrame(result, index=frame.index)
    if not np.isfinite(encoded.to_numpy()).all():
        raise ValueError('Nonfinite feature values.')
    return encoded


def load_main():
    frame = pd.read_csv(ROOT / 'data/main.csv.gz', dtype=str, keep_default_na=False)
    if len(frame) != 15000 or frame['global_record_id'].duplicated().any():
        raise ValueError('Unexpected main records.')
    if frame['split'].value_counts().to_dict() != {'train': 10500, 'test': 3000, 'val': 1500}:
        raise ValueError('Unexpected saved split.')
    frame['source'] = frame['source_file'].map({
        'input.json': 'synthetic', 'Lab dataset.json': 'laboratory'})
    if frame['source'].isna().any():
        raise ValueError('Unknown source.')
    return frame, encode(frame)


def columns(frame, groups):
    return [c for g in groups for c in frame if c.startswith(g + '_')]


def scores(truth, prediction):
    return {
        'accuracy': float(accuracy_score(truth, prediction)),
        'macro_f1': float(f1_score(truth, prediction, labels=np.unique(truth),
                                 average='macro', zero_division=0)),
    }


def source_scores(truth, prediction, sources):
    result = {}
    for source in ['combined', 'laboratory', 'synthetic']:
        mask = np.ones(len(truth), dtype=bool) if source == 'combined' else sources == source
        result[source] = scores(truth[mask], prediction[mask])
    return result


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')
