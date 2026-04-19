# PitchIQ

Football Outcome Intelligence System

# FIFA World Cup Match Outcome Predictor

A machine learning project that predicts the outcome of international football matches — Win, Loss, or Draw — using historical FIFA match data, statistical feature engineering, and interpretable classification models.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Features](#features)
- [Dataset Description](#dataset-description)
- [Methodology](#methodology)
- [Feature Engineering Highlights](#feature-engineering-highlights)
- [Model Performance](#model-performance)
- [Project Structure](#project-structure)
- [Installation and Setup](#installation-and-setup)
- [Usage](#usage)
- [Screenshots and Demo](#screenshots-and-demo)
- [Future Improvements](#future-improvements)
- [Key Learnings](#key-learnings)
- [License](#license)

---

## Project Overview

International football is inherently unpredictable, yet patterns in historical data reveal meaningful signals — teams in strong recent form, favorable head-to-head records, and ranking advantages all correlate with match outcomes. This project builds a systematic, data-driven pipeline to quantify these signals and translate them into probabilistic match predictions.

The predictor ingests over 150 years of international match records, engineers contextually rich features, and trains classification models to estimate the probability of a Win, Loss, or Draw for a given home team. The project is structured in phases to allow incremental improvement, starting from a core statistical model and extending toward player-level and sentiment-enriched predictions.

**Why this project matters:**

- Football analytics is a rapidly growing field, with clubs and federations investing heavily in data infrastructure.
- Public match prediction models are often opaque. This project prioritizes interpretability alongside accuracy.
- The modular architecture makes it straightforward to incorporate richer data sources as they become available.

**What this project achieves:**

- A trained classification model that predicts match outcomes with calibrated probability scores
- A Streamlit web application that allows users to select two national teams and receive a prediction
- SHAP-based feature attribution explaining which factors drove each prediction
- Optuna-powered hyperparameter tuning for optimal model selection
- Probability calibration for reliable confidence estimates

---

## Features

- **Match outcome prediction**: Classifies each match as a Win, Loss, or Draw from the perspective of the home team
- **Calibrated probability outputs**: Returns calibrated confidence scores for all three outcome classes via `CalibratedClassifierCV`
- **SHAP-based explanation**: Surfaces the top contributing features for each prediction with signed SHAP values, color-coded in the UI (teal = pushes toward predicted outcome, orange = pushes against)
- **Hyperparameter tuning**: Optuna-based search across all model types with temporal validation
- **Per-class threshold tuning**: Grid search over per-class probability scaling factors to boost under-predicted classes (especially draws), maximizing macro F1
- **Soft-voting ensemble**: Optional `--ensemble` mode that combines top-performing models via averaged probabilities, used when it beats or ties the best individual model
- **Class imbalance handling**: Balanced class weights for LogisticRegression/RandomForest and balanced sample weights for GradientBoosting/HistGradientBoosting
- **6 model candidates**: Logistic Regression, Random Forest, Gradient Boosting, HistGradientBoosting, XGBoost, and LightGBM (last two optional, require libomp)
- **29 engineered features**: Elo ratings, rolling form (win rate, goals, goal difference), head-to-head records, win/loss streaks, days since last match, and home advantage
- **Modular pipeline**: Each stage — data cleaning, feature engineering, tuning, training, inference — is independently executable
- **Interactive Streamlit interface**: Allows non-technical users to explore predictions through a browser-based UI with prediction cards, SHAP charts, and head-to-head history tables
- **Comprehensive test suite**: 47 pytest tests covering feature engineering (Elo, form, H2H, streaks, days-since-last), model predictions, ensemble behavior, and pipeline integration
- **Player strength integration** *(Phase 2)*: Augments match-level features with squad-level market value and ratings from Transfermarkt
- **Sentiment analysis** *(Phase 3, optional)*: Incorporates pre-match sentiment signals derived from news and social media using transformer-based NLP models

---

## Dataset Description

### 1. FIFA International Match Results

**Source:** [Kaggle — International Football Results from 1872 to 2017](https://www.kaggle.com/datasets/martj42/international-football-results-from-1872-to-2017)

This dataset contains the results of every recorded international men's football match since 1872. It is the foundational data source for the project. A synthetic 500-row sample can be generated for development without Kaggle credentials.

| Column | Description |
|---|---|
| `date` | Date of the match |
| `home_team` | Name of the home team |
| `away_team` | Name of the away team |
| `home_score` | Goals scored by the home team |
| `away_score` | Goals scored by the away team |
| `tournament` | Type of match (e.g., FIFA World Cup, Friendly, AFCON) |
| `city` | City where the match was played |
| `country` | Country where the match was played |
| `neutral` | Boolean flag indicating whether the venue was neutral |

### 2. Transfermarkt Player Data

**Source:** [Kaggle — Player Scores by David Cariboo](https://www.kaggle.com/datasets/davidcariboo/player-scores)

This dataset provides player-level statistics, market values, and club affiliations scraped from Transfermarkt. It is used in Phase 2 to construct squad strength features for each national team.

| Column | Description |
|---|---|
| `player_id` | Unique identifier for each player |
| `name` | Player's full name |
| `market_value_in_eur` | Estimated market value in euros |
| `position` | Player position (e.g., goalkeeper, midfielder) |
| `country_of_citizenship` | National team eligibility |
| `club_name` | Current club affiliation |

### 3. Sentiment Analysis *(Optional — Phase 3)*

**Reference:** [Hugging Face — Sentiment Analysis with Python](https://huggingface.co/blog/sentiment-analysis-python)

Pre-match sentiment is derived from news headlines and social media commentary using a pre-trained transformer model from the Hugging Face ecosystem. Sentiment polarity scores are computed per team and incorporated as additional input features.

---

## Methodology

### 1. Data Cleaning

- Filter out matches with missing scores, ambiguous team names, or corrupted entries
- Standardize team name strings for consistent merging across datasets (via `normalize_team_name()` helper)
- Normalize the `neutral` column from various formats (TRUE/FALSE strings, booleans) to integer
- Remove matches from before a chosen historical cutoff to reduce noise from sparse early records
- Encode the target variable: Win = 1, Draw = 0, Loss = -1 (from the home team perspective)

### 2. Feature Engineering

Raw match records are transformed into a rich feature matrix capturing team form, historical dominance, and contextual match factors. Key engineered features are described in detail in the [Feature Engineering Highlights](#feature-engineering-highlights) section.

### 3. Model Selection

Six classification models are evaluated on a temporal validation split:

- **Logistic Regression** (with StandardScaler) — establishes a probabilistic baseline with interpretable coefficients
- **Random Forest Classifier** — captures non-linear feature interactions with feature importance rankings
- **Gradient Boosting Classifier** — traditional sklearn gradient boosting ensemble
- **HistGradientBoosting Classifier** — sklearn's histogram-based gradient boosting (similar to LightGBM, no native library dependencies)
- **XGBoost** *(optional, requires libomp)* — high-performance gradient boosting with label mapping for non-negative labels
- **LightGBM** *(optional, requires libomp)* — histogram-based gradient boosting from Microsoft

The best model is selected based on macro F1 score on the temporal validation split. Optuna hyperparameter tuning is available via `--tune` to search optimal parameters for each model. With `--ensemble`, a soft-voting ensemble is built from all models scoring within 0.05 macro F1 of the best individual; the ensemble is used only if it beats or ties the best single model.

### 4. Probability Calibration

The selected model is wrapped in `CalibratedClassifierCV` to produce well-calibrated probability estimates:
- **Sigmoid (Platt scaling)** for smaller datasets (<500 samples)
- **Isotonic regression** for larger datasets
- Calibration can be skipped with `--no-calibrate`

### 5. Per-Class Threshold Tuning

After calibration, per-class decision thresholds are optimized on the validation set. Instead of using argmax on raw predicted probabilities, each class's probability is scaled by a learned factor before taking argmax. This allows the model to boost under-predicted classes (especially draws):

- **Grid search** over scaling factors for each class: Loss and Win range [0.5, 1.5], Draw range [0.5, 3.0] (wider because draws are systematically under-predicted)
- The optimal thresholds are saved to `models/thresholds.json` and loaded at inference time by both `predict.py` and `evaluate.py`
- The evaluation report compares raw (argmax) vs. threshold-adjusted metrics side-by-side

### 6. Evaluation

Models are evaluated on a temporally held-out test set (last 10% of matches) to simulate realistic out-of-sample performance. The evaluation report includes both **raw (argmax)** and **threshold-adjusted** predictions:

- **Accuracy** — overall fraction of correctly predicted outcomes (reported for both raw and threshold-adjusted)
- **Macro F1 Score** — accounts for class imbalance across Win, Draw, and Loss (reported for both)
- **Confusion Matrix** — identifies systematic misclassifications between classes (reported for both)
- **Per-class Precision, Recall, F1** — detailed breakdown for each outcome class
- **Macro F1 Delta** — the change in macro F1 from applying threshold tuning
- **Brier Score** — measures calibration quality of predicted probabilities
- **Log Loss** — penalizes overconfident incorrect predictions
- **Reliability Diagram** — per-class calibration curve data (fraction of positives vs. mean predicted value)
- **Calibration Status** — whether the model was probability-calibrated

---

## Feature Engineering Highlights

Feature engineering is the most consequential stage of this pipeline. The following features are constructed for each match:

**Recent Form**
- Rolling win rate over the last 5 and 10 matches for both home and away teams
- Average goals scored and conceded over the same rolling windows
- Average goal difference over the last 5 and 10 matches (`goal_diff_avg_5`, `goal_diff_avg_10`)
- Uses `shift(1).rolling()` to strictly avoid data leakage (only past matches used)

**Momentum and Rest**
- Win/loss streak: positive values indicate consecutive wins, negative values indicate consecutive losses, 0 indicates the last result was a draw
- Days since last match: calendar days since the team last played any match (home or away), default 30 for a team's first match
- Iterates chronologically over a stacked team-level view for correct tracking regardless of home/away role

**Home Advantage**
- Rolling win rate over the team's last 10 home matches (`home_advantage`)
- Only considers matches where the team was the home side
- Captures venue-specific performance trends beyond the binary neutral flag

**Head-to-Head Record**
- Historical win rate of the home team in direct matchups against the away team
- Average goal difference in previous encounters
- Number of prior meetings (used as a confidence weight)
- Uses `tuple(sorted([home, away]))` as a symmetric matchup key

**Ranking and Prestige**
- Elo ratings constructed from the historical match record (base 1500, K=32)
- Elo rating gap between home and away team prior to kickoff (`elo_diff`)
- Ratings updated after each match chronologically

**Venue and Context**
- Whether the match is played at a neutral venue
- Tournament type encoded as a binary feature (`is_friendly`)
- Tournament importance weights available via `tournament_to_weight()` helper

**Squad Strength** *(Phase 2)*
- Aggregate market value of the starting squad derived from Transfermarkt data
- Market value differential between home and away squads
- Positional breakdown: attacking vs. defensive squad value ratio

**Sentiment Score** *(Phase 3, optional)*
- Average sentiment polarity of pre-match news headlines for each team
- Sentiment differential between home and away team coverage

---

## Model Performance

Model evaluation is conducted on a temporally stratified held-out test set. Results are saved to `models/evaluation_report.json` and include:

- Accuracy across Win, Draw, and Loss classes
- Per-class Precision, Recall, and F1 Score
- Macro-averaged F1 Score
- Confusion Matrix
- Log Loss and Brier Score
- Probability calibration status and reliability diagram data
- SHAP-based feature importance ranking

Note: No performance numbers are hardcoded in this README. Actual results depend on the chosen historical cutoff, feature set, and hyperparameter configuration. Run `python -m src.models.evaluate` to see current metrics.

---

## Project Structure

```
PitchIQ/
│
├── data/
│   ├── raw/                    # Original downloaded datasets (not committed to Git)
│   └── processed/              # Cleaned and feature-engineered datasets
│
├── scripts/
│   ├── generate_sample_data.py # Generate synthetic 500-row dataset for development
│   └── download_data.py        # Download real datasets from Kaggle
│
├── src/
│   ├── data/
│   │   ├── loader.py           # Data loading utilities and get_project_root()
│   │   └── cleaner.py          # Data cleaning and target encoding
│   ├── features/
│   │   ├── pipeline.py         # Feature pipeline orchestrator (Elo -> Form -> H2H -> Streak)
│   │   ├── elo.py              # Elo rating computation
│   │   ├── form.py             # Rolling form, goal diff avg, and home advantage
│   │   ├── head_to_head.py     # Head-to-head statistics
│   │   └── streak.py           # Win/loss streaks and days-since-last-match
│   ├── models/
│   │   ├── train.py            # Model training, selection, calibration, and threshold optimization
│   │   ├── tune.py             # Optuna hyperparameter tuning
│   │   ├── threshold.py        # Per-class decision threshold tuning
│   │   ├── ensemble.py         # Soft-voting ensemble of top models
│   │   ├── predict.py          # Inference with SHAP explanations and threshold application
│   │   └── evaluate.py         # Evaluation metrics (raw vs threshold-adjusted)
│   └── utils/
│       └── helpers.py          # Team name normalization, tournament weights
│
├── app/
│   ├── streamlit_app.py        # Main Streamlit application
│   ├── components/
│   │   ├── prediction_card.py  # Outcome badge + probability bar
│   │   ├── feature_attribution.py # SHAP value horizontal bar chart
│   │   └── h2h_table.py        # Head-to-head history table
│   └── assets/                 # Static assets (logos, CSS)
│
├── models/
│   ├── best_model.pkl          # Serialized trained model (gitignored)
│   ├── feature_columns.json    # Feature schema for inference alignment
│   ├── thresholds.json         # Per-class decision thresholds (gitignored)
│   ├── evaluation_report.json  # Latest evaluation metrics (gitignored)
│   └── tuned_params.json       # Optuna tuning results (gitignored)
│
├── tests/
│   ├── conftest.py             # Shared test fixtures (20-match sample)
│   ├── test_features.py        # Tests for Elo, form, H2H, streak, and days-since-last features (27 tests)
│   └── test_models.py          # Tests for predictor API, ensemble behavior, and pipeline integration (20 tests)
│
├── pyproject.toml              # Project config, editable install, pytest config
├── requirements.txt
├── CLAUDE.md
└── README.md
```

---

## Installation and Setup

### Prerequisites

- Python 3.10 or higher
- pip
- Git

### Step 1: Clone the Repository

```bash
git clone https://github.com/your-username/PitchIQ.git
cd PitchIQ
```

### Step 2: Create a Virtual Environment

```bash
python -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate
```

### Step 3: Install the Project

```bash
pip install -e .
```

This installs the project in editable mode with all dependencies. Alternatively:

```bash
pip install -r requirements.txt
```

### Step 4: Get the Data

**Option A: Generate synthetic sample data (quick start, no credentials needed):**

```bash
python scripts/generate_sample_data.py
```

**Option B: Download real Kaggle datasets (~45k matches):**

```bash
python scripts/download_data.py
```

Requires a Kaggle account and [API credentials](https://github.com/Kaggle/kaggle-api#api-credentials).

### Step 5: Run the Pipeline

```bash
python -m src.data.cleaner              # Clean raw data
python -m src.features.pipeline         # Feature engineering
python -m src.models.train              # Train and select best model
python -m src.models.evaluate           # Evaluate on holdout set
```

### Step 6: Run Tests

```bash
pytest tests/ -v
```

### Step 7: Launch the Streamlit App

```bash
streamlit run app/streamlit_app.py
```

The application will open in your browser at `http://localhost:8501`.

---

## Usage

### Using the Streamlit Application

1. Open the app in your browser after running the launch command above.
2. Select the **Home Team** from the dropdown menu.
3. Select the **Away Team** from the second dropdown.
4. Optionally specify the **Tournament Type** (e.g., World Cup, Friendly) to contextualize the prediction.
5. Click **Predict Outcome**.

The app displays:
- A colored outcome badge (green=Win, yellow=Draw, red=Loss) with a stacked probability bar
- A SHAP feature attribution chart showing which factors drove the prediction
- A head-to-head history table with win/draw/loss tallies

### Training with Hyperparameter Tuning

```bash
# Tune all models with Optuna (30 trials each), then train the best
python -m src.models.train --tune

# Tune with more trials
python -m src.models.train --tune --tune-trials 50

# Reuse previously saved tuned parameters
python -m src.models.train --params models/tuned_params.json

# Train without probability calibration
python -m src.models.train --no-calibrate
```

### Ensemble Training

```bash
# Build a soft-voting ensemble from top models
python -m src.models.train --ensemble

# Combine tuning with ensemble selection
python -m src.models.train --tune --ensemble

# Ensemble uses all models within 0.05 macro F1 of the best individual.
# The ensemble is kept only if it beats or ties the best single model.
# Probability calibration is skipped for ensembles (probability averaging
# across diverse model types is self-calibrating).
```

### Standalone Tuning

```bash
# Tune all models and save results
python -m src.models.tune --save

# Tune a specific model
python -m src.models.tune --model HistGradientBoosting --n-trials 50
```

### Threshold Tuning

Per-class decision thresholds are automatically optimized during training. The thresholds are saved to `models/thresholds.json` and applied at inference time. To see the impact of threshold tuning, run:

```bash
python -m src.models.evaluate
# Output includes both raw (argmax) and threshold-adjusted metrics side-by-side
```

---

## Screenshots and Demo

*Screenshots and a live demo link will be added once the Streamlit application is deployed.*

**UI sections:**

- Team selection interface with dropdown menus
- Probability output displayed as a horizontal stacked bar chart (green/yellow/red)
- SHAP feature attribution panel with horizontal bar chart (teal/orange)
- Historical head-to-head summary table with win/draw/loss metrics

---

## Future Improvements

- **Player-level modeling**: Incorporate individual player ratings and injury status to generate a dynamic pre-match squad strength score
- **Sentiment analysis integration**: Use a Hugging Face transformer model to score pre-match press coverage and social media sentiment as additional features
- **Real-time predictions**: Connect to a live football data API to enable predictions for upcoming scheduled fixtures
- **REST API deployment**: Wrap the inference pipeline in a FastAPI service and deploy to a cloud platform (e.g., AWS, Render, Railway)
- **Temporal model updating**: Implement an online learning or periodic retraining schedule to incorporate the most recent match results automatically
- **CI/CD pipeline**: GitHub Actions workflow to run `generate_sample_data -> pipeline -> pytest` on every push
- **Multi-label confidence intervals**: Report uncertainty bounds on predicted probabilities to communicate model confidence

---

## Key Learnings

This project reinforced several principles that apply broadly across applied machine learning work:

**Data quality determines ceiling, not algorithms.** The single most impactful investment was in careful data cleaning and feature construction. Poorly engineered features could not be compensated for by switching to a more complex model.

**Temporal validation is non-negotiable in time-series settings.** Using a random train-test split on match data would artificially inflate performance metrics by allowing the model to train on future information. A temporal cutoff — where the test set consists exclusively of the most recent matches — produces a far more realistic and honest evaluation.

**Interpretability builds trust.** A model that returns "Brazil wins, 62% confidence" is far more useful than a black-box prediction when the user can also see that the Elo rating gap and recent form were the primary drivers. SHAP-based feature attribution is not optional for a sports prediction system — it is the product.

**Probability calibration matters.** Raw model probabilities are often poorly calibrated. Wrapping models in `CalibratedClassifierCV` with isotonic or sigmoid calibration produces more reliable confidence estimates that users can trust.

**Draws are the hardest class to predict.** Class imbalance is severe for Draw outcomes. Addressing this through class weighting, oversampling, or threshold tuning is essential to avoid a model that effectively ignores draws entirely.

**Domain knowledge accelerates progress.** Understanding that Friendly matches carry less competitive signal, or that teams often field weakened squads in non-competitive fixtures, directly informed feature design decisions that would not be obvious from the data alone.

---

*Built as a data science portfolio project. Contributions and feedback are welcome via GitHub Issues or Pull Requests.*
