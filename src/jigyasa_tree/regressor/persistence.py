import joblib
from pathlib import Path

def save_tree(builder, path):
    
    path=Path(path)

    if not path.parent.parent.exists():
        raise FileNotFoundError("path not found")

    path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(builder,path)

def load_tree(path):
    return joblib.load(path)

            

