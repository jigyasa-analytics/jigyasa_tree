import joblib
from pathlib import Path


def save_tree(builder, path):

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(builder, path)


def load_tree(path):
    # joblib uses pickle: only load model files from a source you trust.
    return joblib.load(path)
