import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location("fresh_data", Path(__file__).with_name("prepare_data.py"))
data = importlib.util.module_from_spec(spec)
spec.loader.exec_module(data)


def row(ident, question, answer="a"):
    return {"id": ident, "dataset": ident.split(":")[0], "question": question, "aliases": [answer]}


class FreshDataTests(unittest.TestCase):
    def test_identity_extraction_ignores_predictions_and_outcomes(self):
        a = {"question_id": "nq:1", "question": "Who— IS it?", "response": "trivia:123", "outcome": "correct"}
        b = {**a, "response": "wrong", "outcome": "error"}
        self.assertEqual(data.identities(a), data.identities(b))
        self.assertEqual(data.identities(a), ({"nq:1"}, {"who is it"}))
        self.assertEqual(data.identities({"diagnostic_ids": {"nq": ["nq:2"]}})[0], {"nq:2"})

    def test_excludes_id_and_normalized_question_and_cross_dataset_overlap(self):
        nq = [row("nq:old", "different"), row("nq:new", "Old? Question"), row("nq:1", "shared"), row("nq:2", "fresh nq")]
        trivia = [row("trivia:1", "SHARED!"), row("trivia:2", "fresh trivia"), row("trivia:3", "another trivia")]
        splits, diagnostics, stats = data.select_fresh(trivia, nq, {"nq:old"}, {"old question"}, count=2, diagnostic=1)
        self.assertEqual({r["id"] for r in splits["test_nq"]}, {"nq:1", "nq:2"})
        self.assertEqual({r["id"] for r in splits["test_trivia"]}, {"trivia:2", "trivia:3"})
        data.validate_splits(splits, {"nq:old"}, {"old question"}, diagnostics, count=2, diagnostic=1)
        self.assertEqual(stats["test_trivia"]["duplicate_within_or_across_official_sources"], 1)

    def test_selection_independent_of_input_order_and_answer_content(self):
        nq = [row(f"nq:{i}", f"nq question {i}") for i in range(8)]
        trivia = [row(f"trivia:{i}", f"trivia question {i}") for i in range(8)]
        first, diagnostic_a, _ = data.select_fresh(trivia, nq, set(), set(), count=4, diagnostic=2)
        second, diagnostic_b, _ = data.select_fresh([{**r, "aliases": ["different answer"]} for r in reversed(trivia)], list(reversed(nq)), set(), set(), count=4, diagnostic=2)
        self.assertEqual({k: [r["id"] for r in v] for k, v in first.items()}, {k: [r["id"] for r in v] for k, v in second.items()})
        self.assertEqual(diagnostic_a, diagnostic_b)

    def test_insufficient_fresh_pool_fails_before_writing(self):
        with self.assertRaisesRegex(ValueError, "Only 1 fresh"):
            data.select_fresh([row("trivia:1", "t")], [row("nq:1", "n")], set(), set(), count=2, diagnostic=1)

    def test_validator_rejects_leakage_and_bad_diagnostic(self):
        splits = {"test_nq": [row("nq:1", "n")], "test_trivia": [row("trivia:1", "t")]}
        diagnostic = {name: [rows[0]["id"]] for name, rows in splits.items()}
        with self.assertRaisesRegex(ValueError, "overlap"):
            data.validate_splits(splits, {"nq:1"}, set(), diagnostic, count=1, diagnostic=1)
        with self.assertRaisesRegex(ValueError, "diagnostic"):
            data.validate_splits(splits, set(), set(), {**diagnostic, "test_nq": ["nq:missing"]}, count=1, diagnostic=1)


if __name__ == "__main__":
    unittest.main()
