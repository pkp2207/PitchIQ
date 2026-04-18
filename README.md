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

- A trained classification model that predicts match outcomes with probability scores
- A Streamlit web application that allows users to select two national teams and receive a prediction
- Clear feature attribution explaining which factors drove the prediction

---

## Features

- **Match outcome prediction**: Classifies each match as a Win, Loss, or Draw from the perspective of the home team
- **Probability outputs**: Returns confidence scores for all three outcome classes, not just a single label
- **Feature-based explanation**: Surfaces the top contributing features for each prediction, making the model interpretable
- **Modular pipeline**: Each stage — data cleaning, feature engineering, training, inference — is independently executable
- **Interactive Streamlit interface**: Allows non-technical users to explore predictions through a browser-based UI
- **Player strength integration** *(Phase 2)*: Augments match-level features with squad-level market value and ratings from Transfermarkt
- **Sentiment analysis** *(Phase 3, optional)*: Incorporates pre-match sentiment signals derived from news and social media using transformer-based NLP models

---

## Dataset Description

### 1. FIFA International Match Results

**Source:** [Kaggle — International Football Results from 1872 to 2017](https://www.kaggle.com/datasets/martj42/international-football-results-from-1872-to-2017)

This dataset contains the results of every recorded international men's football match since 1872. It is the foundational data source for the project.

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
- Standardize team name strings for consistent merging across datasets
- Remove matches from before a chosen historical cutoff to reduce noise from sparse early records
- Encode the target variable: Win = 1, Draw = 0, Loss = -1 (from the home team perspective)

### 2. Feature Engineering

Raw match records are transformed into a rich feature matrix capturing team form, historical dominance, and contextual match factors. Key engineered features are described in detail in the [Feature Engineering Highlights](#feature-engineering-highlights) section.

### 3. Model Selection

Multiple classification models are evaluated:

- **Logistic Regression** — establishes a probabilistic baseline and provides directly interpretable coefficients
- **Random Forest Classifier** — captures non-linear feature interactions and provides feature importance rankings
- **Gradient Boosting (XGBoost / LightGBM)** — typically achieves the strongest predictive performance through ensemble boosting

The final model is selected based on cross-validated F1 score and calibration of predicted probabilities.

### 4. Evaluation

Models are evaluated on a temporally held-out test set (matches from the most recent years) to simulate realistic out-of-sample performance. The following metrics are reported:

- **Accuracy** — overall fraction of correctly predicted outcomes
- **Macro F1 Score** — accounts for class imbalance across Win, Draw, and Loss
- **Confusion Matrix** — identifies systematic misclassifications between classes
- **Brier Score** — measures calibration quality of predicted probabilities
- **Log Loss** — penalizes overconfident incorrect predictions

---

## Feature Engineering Highlights

Feature engineering is the most consequential stage of this pipeline. The following features are constructed for each match:

**Recent Form**
- Rolling win rate, draw rate, and loss rate over the last 5 and 10 matches for both home and away teams
- Average goals scored and conceded over the same rolling windows
- Momentum indicators based on whether the team is on a winning or losing streak

**Head-to-Head Record**
- Historical win rate of the home team in direct matchups against the away team
- Average goal difference in previous encounters
- Number of prior meetings (used as a confidence weight)

**Ranking and Prestige**
- FIFA ranking differential at the time of the match (where available)
- Proxy ranking based on Elo ratings, constructed from the historical match record
- Elo rating gap between home and away team prior to kickoff

**Venue and Context**
- Whether the match is played at a neutral venue
- Tournament type encoded as an ordinal feature (Friendlies carry less signal than competitive fixtures)
- Home advantage indicator (non-neutral ground matches)

**Squad Strength** *(Phase 2)*
- Aggregate market value of the starting squad derived from Transfermarkt data
- Market value differential between home and away squads
- Positional breakdown: attacking vs. defensive squad value ratio

**Sentiment Score** *(Phase 3, optional)*
- Average sentiment polarity of pre-match news headlines for each team
- Sentiment differential between home and away team coverage

---

## Model Performance

Model evaluation is conducted on a temporally stratified held-out test set. The following metrics are tracked and reported in the `notebooks/04_model_evaluation.ipynb` notebook:

- Accuracy across Win, Draw, and Loss classes
- Per-class Precision, Recall, and F1 Score
- Macro-averaged and weighted F1 Score
- Confusion Matrix visualization
- Probability calibration curve (reliability diagram)
- Feature importance ranking (for tree-based models)

Note: No performance numbers are hardcoded in this README. Actual results depend on the chosen historical cutoff, feature set, and hyperparameter configuration. Refer to the evaluation notebook for reported metrics.

---

## Project Structure

```
fifa-match-predictor/
│
├── data/
│   ├── raw/                    # Original downloaded datasets (not committed to Git)
│   ├── processed/              # Cleaned and feature-engineered datasets
│   └── external/               # Supplementary data (Transfermarkt, rankings)
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_model_training.ipynb
│   └── 04_model_evaluation.ipynb
│
├── src/
│   ├── data/
│   │   ├── loader.py           # Data loading utilities
│   │   └── cleaner.py          # Data cleaning functions
│   ├── features/
│   │   ├── form.py             # Rolling form feature computation
│   │   ├── head_to_head.py     # Head-to-head statistics
│   │   ├── elo.py              # Elo rating computation
│   │   └── squad_strength.py   # Transfermarkt integration (Phase 2)
│   ├── models/
│   │   ├── train.py            # Model training and cross-validation
│   │   ├── predict.py          # Inference utilities
│   │   └── evaluate.py         # Evaluation metric computation
│   └── utils/
│       └── helpers.py          # Shared utility functions
│
├── app/
│   ├── streamlit_app.py        # Main Streamlit application
│   ├── components/             # UI component modules
│   └── assets/                 # Static assets (logos, CSS)
│
├── models/
│   ├── best_model.pkl          # Serialized trained model
│   └── feature_columns.json    # Feature schema for inference alignment
│
├── tests/
│   ├── test_features.py
│   └── test_models.py
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Installation and Setup

### Prerequisites

- Python 3.9 or higher
- pip
- Git

### Step 1: Clone the Repository

```bash
git clone https://github.com/your-username/fifa-match-predictor.git
cd fifa-match-predictor
```

### Step 2: Create a Virtual Environment

```bash
python -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Download the Datasets

Download the following datasets from Kaggle and place them in the `data/raw/` directory:

- [International Football Results](https://www.kaggle.com/datasets/martj42/international-football-results-from-1872-to-2017) — save as `data/raw/results.csv`
- [Transfermarkt Player Scores](https://www.kaggle.com/datasets/davidcariboo/player-scores) — save as `data/raw/players.csv`

You will need a Kaggle account and the [Kaggle API](https://github.com/Kaggle/kaggle-api) configured to download via CLI:

```bash
kaggle datasets download -d martj42/international-football-results-from-1872-to-2017 -p data/raw/ --unzip
kaggle datasets download -d davidcariboo/player-scores -p data/raw/ --unzip
```

### Step 5: Run the Data Pipeline

```bash
python src/data/cleaner.py
python src/features/form.py
python src/features/elo.py
```

### Step 6: Train the Model

```bash
python src/models/train.py
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
5. Click **Predict**.

**Example Output:**

```
Home Team: Brazil
Away Team: Argentina
Tournament: FIFA World Cup

Predicted Outcome: Win (Home)

Probabilities:
  Win  (Brazil):    62.4%
  Draw:             21.1%
  Loss (Brazil):    16.5%

Top Contributing Features:
  1. Elo Rating Differential:         +0.38
  2. Home Team Recent Win Rate (5):   +0.27
  3. Head-to-Head Win Rate:           +0.19
  4. Neutral Venue:                   -0.08
  5. Away Team Recent Form (10):      -0.06
```

Note: The probability values shown above are illustrative examples for documentation purposes only. Actual model outputs will vary based on the training data and configuration.

### Running Notebooks

Notebooks are intended to be run sequentially:

```bash
jupyter notebook notebooks/01_data_exploration.ipynb
```

---

## Screenshots and Demo

*Screenshots and a live demo link will be added once the Streamlit application is deployed.*

**Planned UI sections:**

- Team selection interface with national team flags
- Probability output displayed as a horizontal bar chart
- Feature attribution panel showing SHAP values or coefficient contributions
- Historical head-to-head summary table

---

## Future Improvements

- **Player-level modeling**: Incorporate individual player ratings and injury status to generate a dynamic pre-match squad strength score
- **Sentiment analysis integration**: Use a Hugging Face transformer model to score pre-match press coverage and social media sentiment as additional features
- **Real-time predictions**: Connect to a live football data API to enable predictions for upcoming scheduled fixtures
- **REST API deployment**: Wrap the inference pipeline in a FastAPI service and deploy to a cloud platform (e.g., AWS, Render, Railway)
- **Temporal model updating**: Implement an online learning or periodic retraining schedule to incorporate the most recent match results automatically
- **Explainability dashboard**: Integrate SHAP-based explanations more deeply into the Streamlit UI for interactive feature attribution
- **Multi-label confidence intervals**: Report uncertainty bounds on predicted probabilities to communicate model confidence

---

## Key Learnings

This project reinforced several principles that apply broadly across applied machine learning work:

**Data quality determines ceiling, not algorithms.** The single most impactful investment was in careful data cleaning and feature construction. Poorly engineered features could not be compensated for by switching to a more complex model.

**Temporal validation is non-negotiable in time-series settings.** Using a random train-test split on match data would artificially inflate performance metrics by allowing the model to train on future information. A temporal cutoff — where the test set consists exclusively of the most recent matches — produces a far more realistic and honest evaluation.

**Interpretability builds trust.** A model that returns "Brazil wins, 62% confidence" is far more useful than a black-box prediction when the user can also see that the Elo rating gap and recent form were the primary drivers. Feature attribution is not optional for a sports prediction system — it is the product.

**Draws are the hardest class to predict.** Class imbalance is severe for Draw outcomes. Addressing this through class weighting, oversampling, or threshold tuning is essential to avoid a model that effectively ignores draws entirely.

**Domain knowledge accelerates progress.** Understanding that Friendly matches carry less competitive signal, or that teams often field weakened squads in non-competitive fixtures, directly informed feature design decisions that would not be obvious from the data alone.

---

*Built as a data science portfolio project. Contributions and feedback are welcome via GitHub Issues or Pull Requests.*