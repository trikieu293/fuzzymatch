"""Unit tests for fuzzymatch.decisions module."""

from __future__ import annotations

from fuzzymatch.decisions import Explanation, FieldScore, MatchDecision


class TestFieldScore:
    def test_to_dict_exact(self) -> None:
        """FieldScore.to_dict() produces the expected dict for a name score."""
        fs = FieldScore(
            field="name",
            left_value="Office Supplies Exp.",
            right_value="Office Supplies Expense",
            score=0.93,
        )
        result = fs.to_dict()
        assert result == {
            "field": "name",
            "left_value": "Office Supplies Exp.",
            "right_value": "Office Supplies Expense",
            "score": 0.93,
        }


class TestMatchDecision:
    def test_nested_serialization(self) -> None:
        """MatchDecision.to_dict() nests Explanation and FieldScore correctly."""
        name_fs = FieldScore(
            field="name",
            left_value="ACME Corp.",
            right_value="ACME Corporation",
            score=0.88,
        )
        explanation = Explanation(
            field_scores=(name_fs,),
            threshold_band="possible",
            blocking_rules=("ExactBlock(country)",),
            classifier_weights={"name": 2.5},
        )
        md = MatchDecision(
            left_id="left-1",
            right_id="right-7",
            status="possible",
            score=0.74,
            explanation=explanation,
        )
        result = md.to_dict()
        assert result == {
            "left_id": "left-1",
            "right_id": "right-7",
            "status": "possible",
            "score": 0.74,
            "explanation": {
                "field_scores": [
                    {
                        "field": "name",
                        "left_value": "ACME Corp.",
                        "right_value": "ACME Corporation",
                        "score": 0.88,
                    }
                ],
                "threshold_band": "possible",
                "blocking_rules": ["ExactBlock(country)"],
                "classifier_weights": {"name": 2.5},
            },
        }
