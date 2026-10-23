# Pre-Connection_BLE_IoT_Device-Type_Identification

Reproducibility experiments for our paper *Toward Lightweight Pre-Connection BLE IoT Device-Type
Identification*.

## Run

Use Python 3.12 in a virtual environment, then run:

```sh
python -m pip install -r requirements.txt
python reproduce.py
```

This runs the classifiers, validation-based feature selection, RF100 removal
and combination experiments, and four-category laboratory evaluation. It then
checks the scores against `reference.json` from the original runs.

For a single experiment, use `python reproduce.py main`, `dependence`, or
`laboratory`. Use `python reproduce.py check` to check existing results.

Results are written to `results/` as JSON. The removal experiment also produces
Figure 3; the model files used for storage measurements are saved in
`results/models/`. Rerun times and file sizes can differ from the recorded Colab
measurements in `reference.json`.

## Data

`data/main.csv.gz` contains the 15,000 records, original labels, saved split,
and 14 retained raw broadcast fields. `data.py` converts them to the 19
float32 classifier coordinates. Unused identifiers and connected-service data
are omitted. These are stored descriptors, not packet captures.

`data/laboratory.csv.gz` contains the separate evaluation's 5,000 laboratory
records and 272 matched synthetic records, eight structural descriptors, and
saved laboratory split. Its preprocessing preserves the original one-hot
entry-type encoding and constant availability indicators.

The two experiments use different saved synthetic inputs. Their source hashes
and extraction notes are in `data/provenance.json`. No records or partitions
are newly generated.

All main test vectors occur in training. These scores measure known-profile
recognition, not independent-device generalization. The laboratory holdout
tests contain only four profiles per source. Synthetic length inconsistencies
are retained rather than corrected.

The original package has no assigned publication license; see `LICENSE`.

