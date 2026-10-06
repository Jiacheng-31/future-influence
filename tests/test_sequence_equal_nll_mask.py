#!/usr/bin/env python3

from pathlib import Path
import sys
import unittest


ENTROPY_DIR = Path(__file__).resolve().parents[1] / "token_selection" / "entropy"
sys.path.insert(0, str(ENTROPY_DIR))

from materialize_sequence_equal_nll_mask import sequence_equal_nll_mask  # noqa: E402


class SequenceEqualNllMaskTest(unittest.TestCase):
    def test_matches_variable_reference_budget_and_masks_lowest_nll(self) -> None:
        mask = sequence_equal_nll_mask(
            [4.0, 1.0, 3.0, 2.0],
            [2.0, -1.0, 0.0, 3.0],
        )
        self.assertEqual(mask, [1.0, 0.0, 1.0, 0.0])

    def test_ties_use_earliest_response_positions(self) -> None:
        mask = sequence_equal_nll_mask(
            [1.0, 1.0, 1.0, 2.0],
            [-1.0, 1.0, 1.0, 1.0],
        )
        self.assertEqual(mask, [0.0, 1.0, 1.0, 1.0])

    def test_zero_and_full_budgets(self) -> None:
        self.assertEqual(
            sequence_equal_nll_mask([1.0, 2.0], [1.0, 2.0]),
            [1.0, 1.0],
        )
        self.assertEqual(
            sequence_equal_nll_mask([1.0, 2.0], [0.0, -1.0]),
            [0.0, 0.0],
        )

    def test_rejects_negative_nll(self) -> None:
        with self.assertRaisesRegex(ValueError, "non-negative"):
            sequence_equal_nll_mask([-0.1], [1.0])


if __name__ == "__main__":
    unittest.main()
