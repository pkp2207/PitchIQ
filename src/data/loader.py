import pandas as pd
from pathlib import Path

def get_project_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent

def load_results() -> pd.DataFrame:
    """Loads the international football results dataset."""
    root = get_project_root()
    path1 = root / "data" / "raw" / "results.csv"
    path2 = root / "dataset" / "results.csv"
    
    if path1.exists():
        path = path1
    elif path2.exists():
        path = path2
    else:
        raise FileNotFoundError(f"Dataset not found at {path1} or {path2}. Please place results.csv there.")
    
    df = pd.read_csv(path)
    df['date'] = pd.to_datetime(df['date'])
    return df

def load_players() -> pd.DataFrame:
    """Loads the Transfermarkt players dataset."""
    root = get_project_root()
    path = root / "data" / "raw" / "players.csv"
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at {path}. Please run scripts/download_data.py first.")
    
    return pd.read_csv(path)
