import pytest
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from src.models.ensemble import SoftVotingEnsemble
from src.models.predict import MatchPredictor


@pytest.fixture(scope="module")
def predictor():
    """Load the trained model predictor (requires pipeline to have been run)."""
    try:
        return MatchPredictor()
    except FileNotFoundError:
        pytest.skip("Model files not found — run the pipeline first.")


class TestMatchPredictor:
    def test_predict_returns_expected_keys(self, predictor):
        result = predictor.predict_match("Brazil", "Argentina")
        assert "outcome" in result
        assert "probabilities" in result
        assert "top_features" in result

    def test_outcome_is_valid(self, predictor):
        result = predictor.predict_match("Brazil", "Argentina")
        assert result["outcome"] in ("Win", "Draw", "Loss")

    def test_probabilities_sum_to_one(self, predictor):
        result = predictor.predict_match("Brazil", "Argentina")
        total = sum(result["probabilities"].values())
        assert total == pytest.approx(1.0, abs=0.01)

    def test_probabilities_keys(self, predictor):
        result = predictor.predict_match("Brazil", "Argentina")
        assert set(result["probabilities"].keys()) == {"Win", "Draw", "Loss"}

    def test_top_features_have_shap_value(self, predictor):
        result = predictor.predict_match("Brazil", "Argentina")
        for feat in result["top_features"]:
            assert "feature" in feat
            assert "shap_value" in feat

    def test_top_features_limited_to_5(self, predictor):
        result = predictor.predict_match("Brazil", "Argentina")
        assert len(result["top_features"]) <= 5

    def test_h2h_history_returns_list(self, predictor):
        records = predictor.get_h2h_history("Brazil", "Argentina")
        assert isinstance(records, list)

    def test_different_matchups_work(self, predictor):
        result = predictor.predict_match("France", "Germany", "Competitive")
        assert result["outcome"] in ("Win", "Draw", "Loss")


class TestPipelineSmoke:
    """Smoke test: ensure the full pipeline produces consistent artifacts."""

    def test_feature_columns_match(self, predictor):
        """Feature columns in JSON should match what the model expects."""
        import json
        from src.data.loader import get_project_root

        root = get_project_root()
        with open(root / "models" / "feature_columns.json") as f:
            cols = json.load(f)

        assert len(cols) == len(predictor.feature_cols)
        assert cols == predictor.feature_cols


# --- SoftVotingEnsemble unit tests ---

@pytest.fixture
def trained_ensemble():
    """Build a small SoftVotingEnsemble from two fitted toy models."""
    rng = np.random.RandomState(42)
    X = rng.randn(200, 4)
    y = np.array([-1, 0, 1] * 66 + [-1, 0])  # length 200, balanced-ish

    rf = RandomForestClassifier(n_estimators=10, random_state=42)
    rf.fit(X, y)

    gb = GradientBoostingClassifier(n_estimators=10, random_state=42)
    gb.fit(X, y)

    ens = SoftVotingEnsemble([("RF", rf), ("GB", gb)])
    return ens, X, y


class TestSoftVotingEnsemble:
    def test_predict_proba_shape(self, trained_ensemble):
        ens, X, _ = trained_ensemble
        proba = ens.predict_proba(X)
        assert proba.shape == (len(X), 3)

    def test_predict_proba_sums_to_one(self, trained_ensemble):
        ens, X, _ = trained_ensemble
        proba = ens.predict_proba(X)
        row_sums = proba.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-6)

    def test_predict_returns_valid_classes(self, trained_ensemble):
        ens, X, _ = trained_ensemble
        preds = ens.predict(X)
        assert set(preds).issubset({-1, 0, 1})

    def test_predict_matches_argmax(self, trained_ensemble):
        """predict() should return the class with the highest averaged probability."""
        ens, X, _ = trained_ensemble
        proba = ens.predict_proba(X)
        expected = ens.classes_[np.argmax(proba, axis=1)]
        np.testing.assert_array_equal(ens.predict(X), expected)

    def test_classes_attribute(self, trained_ensemble):
        ens, _, _ = trained_ensemble
        assert hasattr(ens, 'classes_')
        assert set(ens.classes_) == {-1, 0, 1}

    def test_feature_importances_shape(self, trained_ensemble):
        ens, X, _ = trained_ensemble
        imp = ens.feature_importances_
        assert imp is not None
        assert imp.shape == (X.shape[1],)

    def test_feature_importances_averages_correctly(self, trained_ensemble):
        ens, _, _ = trained_ensemble
        # Manually average the two sub-model importances
        rf_imp = ens.estimators_[0].feature_importances_
        gb_imp = ens.estimators_[1].feature_importances_
        expected = (rf_imp + gb_imp) / 2
        np.testing.assert_allclose(ens.feature_importances_, expected)

    def test_feature_importances_skips_non_tree_models(self):
        """If an LR pipeline is in the ensemble, it should be skipped for importances."""
        rng = np.random.RandomState(42)
        X = rng.randn(200, 4)
        y = np.array([-1, 0, 1] * 66 + [-1, 0])

        lr = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, random_state=42)),
        ])
        lr.fit(X, y)

        rf = RandomForestClassifier(n_estimators=10, random_state=42)
        rf.fit(X, y)

        ens = SoftVotingEnsemble([("LR", lr), ("RF", rf)])
        imp = ens.feature_importances_
        # Only RF contributes, so importances should equal RF's
        np.testing.assert_allclose(imp, rf.feature_importances_)

    def test_feature_importances_none_if_no_tree_models(self):
        """Ensemble of only linear models should return None for importances."""
        rng = np.random.RandomState(42)
        X = rng.randn(200, 4)
        y = np.array([-1, 0, 1] * 66 + [-1, 0])

        lr1 = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, random_state=42)),
        ])
        lr1.fit(X, y)

        lr2 = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(C=0.1, max_iter=1000, random_state=42)),
        ])
        lr2.fit(X, y)

        ens = SoftVotingEnsemble([("LR1", lr1), ("LR2", lr2)])
        assert ens.feature_importances_ is None

    def test_empty_estimators_raises(self):
        with pytest.raises(ValueError, match="must not be empty"):
            SoftVotingEnsemble([])

    def test_repr(self, trained_ensemble):
        ens, _, _ = trained_ensemble
        assert "RF" in repr(ens)
        assert "GB" in repr(ens)
