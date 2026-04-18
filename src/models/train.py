import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import json
import joblib
import pandas as pd
from pathlib import Path
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from src.data.loader import get_project_root

def train_and_select_model():
    root = get_project_root()
    data_path = root / "data" / "processed" / "features.csv"
    
    if not data_path.exists():
        raise FileNotFoundError(f"Features file not found at {data_path}. Run src/features/pipeline.py first.")
        
    df = pd.read_csv(data_path)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    
    # Exclude non-numeric and target columns from features
    exclude_cols = ['date', 'home_team', 'away_team', 'tournament', 'outcome']
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    
    X = df[feature_cols]
    y = df['outcome']
    
    # Train-test split based on time (last 10% of data as holdout test)
    split_idx = int(len(df) * 0.9)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    print(f"Training set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    
    # Initialize models
    models = {
        'LogisticRegression': LogisticRegression(max_iter=1000, random_state=42),
        'RandomForest': RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    }
    
    best_model_name = None
    best_score = -1
    best_model = None
    
    # Evaluate models on validation set (we'll just use a simple time split within train set for validation)
    val_split_idx = int(len(X_train) * 0.8)
    X_t, X_v = X_train.iloc[:val_split_idx], X_train.iloc[val_split_idx:]
    y_t, y_v = y_train.iloc[:val_split_idx], y_train.iloc[val_split_idx:]
    
    print("\nModel Evaluation (Macro F1 on validation set):")
    for name, model in models.items():
        model.fit(X_t, y_t)
        preds = model.predict(X_v)
        score = f1_score(y_v, preds, average='macro')
        print(f"{name}: {score:.4f}")
        
        if score > best_score:
            best_score = score
            best_model_name = name
            best_model = model
            
    print(f"\nBest Model: {best_model_name} (Retraining on full training set...)")
    
    # Retrain best model on full training set
    best_model.fit(X_train, y_train)
    
    # Save the model and feature columns
    out_dir = root / "models"
    os.makedirs(out_dir, exist_ok=True)
    
    model_path = out_dir / "best_model.pkl"
    joblib.dump(best_model, model_path)
    
    cols_path = out_dir / "feature_columns.json"
    with open(cols_path, 'w') as f:
        json.dump(feature_cols, f)
        
    print(f"Model saved to {model_path}")
    print(f"Feature columns saved to {cols_path}")

if __name__ == "__main__":
    train_and_select_model()
