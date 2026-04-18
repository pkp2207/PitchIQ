import pytest
import numpy as np
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
