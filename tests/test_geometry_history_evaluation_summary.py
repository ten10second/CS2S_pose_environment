import json
import tempfile
import unittest
from pathlib import Path

from tools.summarize_geometry_history_evaluation import EvaluationSummaryError, output_path, summarize_evaluation


CONDITIONS = ("disabled", "correct", "wrong_history", "wrong_geometry")


def write_jsonl(path, rows):
    with Path(path).open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def make_row(
    pair_id,
    drive,
    frame_index,
    timestep,
    condition,
    all_loss,
    valid_loss,
    invalid_loss,
    *,
    cohort="expanded",
    depth=0.01,
    valid_fraction=0.25,
    valid_cells=25,
    total_cells=100,
):
    row = {
        "cohort": cohort,
        "split": "val",
        "pair_id": pair_id,
        "drive": drive,
        "frame_index": frame_index,
        "t": timestep,
        "condition": condition,
        "loss_eps_base": all_loss,
        "loss_lidar_bottleneck_depth_log_l1": depth,
        "region_eps": {
            "all": all_loss,
            "valid": valid_loss,
            "invalid": invalid_loss,
            "valid_fraction": valid_fraction,
            "valid_cells": valid_cells,
            "total_cells": total_cells,
        },
        "attention": [
            {
                "block_index": 12,
                "query_valid_fraction": valid_fraction,
                "null_all": 0.2,
                "null_valid": 0.1,
                "residual_to_x_all": 0.01,
                "residual_to_x_valid": 0.02,
                "residual_to_cond_all": 0.03,
            }
        ],
    }
    if condition == "disabled":
        row["disabled_repeat_equal"] = True
    return row


def condition_rows(
    pair_id,
    drive,
    frame_index,
    timestep,
    *,
    cohort="expanded",
    disabled=(1.0, 1.0, 1.0),
    correct=(0.99, 0.80, 1.05),
    wrong_history=(1.01, 0.90, 1.08),
    wrong_geometry=(1.02, 0.95, 1.10),
    depth=0.01,
    valid_fraction=0.25,
):
    losses = {
        "disabled": disabled,
        "correct": correct,
        "wrong_history": wrong_history,
        "wrong_geometry": wrong_geometry,
    }
    rows = []
    for condition in CONDITIONS:
        rows.append(
            make_row(
                pair_id,
                drive,
                frame_index,
                timestep,
                condition,
                *losses[condition],
                cohort=cohort,
                depth=depth,
                valid_fraction=valid_fraction,
            )
        )
    return rows


class GeometryHistoryEvaluationSummaryTests(unittest.TestCase):
    def make_eval_dir(self, rows, selection_ids, *, timesteps=(250,), world_size=1, original_ids=None):
        tmp = tempfile.TemporaryDirectory()
        root = Path(tmp.name)
        selection = {
            "expanded": [
                {"pair_id": pair_id, "drive": "drive", "frame_index": i}
                for i, pair_id in enumerate(selection_ids)
            ],
            "world_size": world_size,
            "timesteps": list(timesteps),
            "conditions": list(CONDITIONS),
        }
        if original_ids is not None:
            selection["original"] = [
                {"pair_id": pair_id, "drive": "original_drive", "frame_index": i}
                for i, pair_id in enumerate(original_ids)
            ]
        (root / "selection.json").write_text(json.dumps(selection))
        for rank in range(world_size):
            (root / f"reproduction_rank{rank}.json").write_text(json.dumps({"passed": True}))
            (root / f"expanded_complete_rank{rank}.json").write_text(json.dumps({"passed": True}))
        expanded = [row for row in rows if row.get("cohort") != "original"]
        original = [row for row in rows if row.get("cohort") == "original"]
        if expanded:
            write_jsonl(root / "evaluation_rank0.jsonl", expanded)
        if original:
            write_jsonl(root / "original_rank0.jsonl", original)
        return tmp, root

    def test_region_summary_exposes_valid_area_gain_hidden_by_full_frame(self):
        rows = condition_rows("p1", "drive_a", 10, 250)
        tmp, root = self.make_eval_dir(rows, ["p1"])
        self.addCleanup(tmp.cleanup)

        summary = summarize_evaluation(root, bootstrap_samples=20, seed=7)
        macro = summary["aggregates"]["expanded"]["250"]["pooled_macro_equal_drive"]
        all_gain = macro["gains"]["all"]["disabled_minus_correct"]
        valid_gain = macro["gains"]["valid"]["disabled_minus_correct"]
        invalid_gain = macro["gains"]["invalid"]["disabled_minus_correct"]

        self.assertAlmostEqual(all_gain["mean_abs"], 0.01)
        self.assertAlmostEqual(valid_gain["mean_abs"], 0.20)
        self.assertAlmostEqual(invalid_gain["mean_abs"], -0.05)
        self.assertEqual(valid_gain["wins"], 1)
        self.assertEqual(invalid_gain["wins"], 0)
        self.assertIn("degenerate", macro["bootstrap_note"])

    def test_macro_equal_drive_does_not_collapse_to_pair_pooled_mean(self):
        rows = []
        for i in range(4):
            rows.extend(
                condition_rows(
                    f"a{i}",
                    "drive_many",
                    100 + i,
                    250,
                    disabled=(1.0, 1.0, 1.0),
                    correct=(0.90, 0.90, 0.90),
                )
            )
        rows.extend(
            condition_rows(
                "b0",
                "drive_one",
                500,
                250,
                disabled=(1.0, 1.0, 1.0),
                correct=(1.20, 1.20, 1.20),
            )
        )
        tmp, root = self.make_eval_dir(rows, ["a0", "a1", "a2", "a3", "b0"])
        self.addCleanup(tmp.cleanup)

        summary = summarize_evaluation(root, bootstrap_samples=10, seed=0)
        payload = summary["aggregates"]["expanded"]["250"]
        micro = payload["pooled_micro"]["gains"]["all"]["disabled_minus_correct"]
        macro = payload["pooled_macro_equal_drive"]["gains"]["all"]["disabled_minus_correct"]

        self.assertAlmostEqual(micro["mean_abs"], 0.04)
        self.assertAlmostEqual(macro["mean_abs"], -0.05)

    def test_original_and_expanded_files_are_reported_separately(self):
        rows = []
        rows.extend(condition_rows("p1", "drive_a", 10, 250))
        rows.extend(condition_rows("o1", "drive_old", 20, 250, cohort="original"))
        tmp, root = self.make_eval_dir(rows, ["p1"])
        self.addCleanup(tmp.cleanup)

        summary = summarize_evaluation(root, bootstrap_samples=5, seed=2)
        self.assertIn("expanded", summary["aggregates"])
        self.assertIn("original", summary["aggregates"])
        self.assertTrue(summary["original_reproduction"]["present"])
        self.assertEqual(summary["selection_coverage"]["seen_expanded_pairs"], 1)

    def test_incomplete_or_duplicate_groups_fail_fast(self):
        rows = condition_rows("p1", "drive_a", 10, 250)
        rows = [row for row in rows if row["condition"] != "wrong_geometry"]
        tmp, root = self.make_eval_dir(rows, ["p1"])
        self.addCleanup(tmp.cleanup)
        with self.assertRaisesRegex(EvaluationSummaryError, "incomplete condition"):
            summarize_evaluation(root)

        dup_rows = condition_rows("p1", "drive_a", 10, 250)
        dup_rows.append(dup_rows[0].copy())
        tmp2, root2 = self.make_eval_dir(dup_rows, ["p1"])
        self.addCleanup(tmp2.cleanup)
        with self.assertRaisesRegex(EvaluationSummaryError, "duplicate"):
            summarize_evaluation(root2)

    def test_depth_and_mask_violations_fail_fast(self):
        rows = condition_rows("p1", "drive_a", 10, 250)
        rows[1]["loss_lidar_bottleneck_depth_log_l1"] = 0.02
        tmp, root = self.make_eval_dir(rows, ["p1"])
        self.addCleanup(tmp.cleanup)
        with self.assertRaisesRegex(EvaluationSummaryError, "depth loss changed"):
            summarize_evaluation(root)

        rows = condition_rows("p1", "drive_a", 10, 250)
        rows[2]["region_eps"]["valid_cells"] = 20
        rows[2]["region_eps"]["valid_fraction"] = 0.2
        tmp2, root2 = self.make_eval_dir(rows, ["p1"])
        self.addCleanup(tmp2.cleanup)
        with self.assertRaisesRegex(EvaluationSummaryError, "inconsistent valid masks"):
            summarize_evaluation(root2)

    def test_null_regions_are_skipped_for_that_region_only(self):
        rows = condition_rows("p1", "drive_a", 10, 250)
        for row in rows:
            row["region_eps"]["valid_fraction"] = 0.0
            row["region_eps"]["valid_cells"] = 0
            row["region_eps"]["valid"] = None
        tmp, root = self.make_eval_dir(rows, ["p1"])
        self.addCleanup(tmp.cleanup)

        summary = summarize_evaluation(root, bootstrap_samples=5, seed=3)
        gains = summary["aggregates"]["expanded"]["250"]["pooled_micro"]["gains"]
        self.assertEqual(gains["valid"]["disabled_minus_correct"]["n"], 0)
        self.assertEqual(gains["all"]["disabled_minus_correct"]["n"], 1)

    def test_bootstrap_is_deterministic(self):
        rows = []
        for i, frame in enumerate((10, 120, 240, 360)):
            rows.extend(
                condition_rows(
                    f"p{i}",
                    "drive_a",
                    frame,
                    250,
                    disabled=(1.0, 1.0, 1.0),
                    correct=(0.90 + i * 0.01, 0.90 + i * 0.01, 0.90 + i * 0.01),
                )
            )
        tmp, root = self.make_eval_dir(rows, ["p0", "p1", "p2", "p3"])
        self.addCleanup(tmp.cleanup)

        first = summarize_evaluation(root, bootstrap_samples=25, seed=42)
        second = summarize_evaluation(root, bootstrap_samples=25, seed=42)
        first_ci = first["aggregates"]["expanded"]["250"]["pooled_micro"]["gains"]["all"][
            "disabled_minus_correct"
        ]["bootstrap"]
        second_ci = second["aggregates"]["expanded"]["250"]["pooled_micro"]["gains"]["all"][
            "disabled_minus_correct"
        ]["bootstrap"]
        self.assertEqual(first_ci, second_ci)
        self.assertFalse(first_ci["degenerate"])

    def test_selected_pairs_require_every_declared_timestep(self):
        rows = condition_rows("p1", "drive_a", 10, 250)
        tmp, root = self.make_eval_dir(rows, ["p1"], timesteps=(250, 750))
        self.addCleanup(tmp.cleanup)
        with self.assertRaisesRegex(EvaluationSummaryError, "selection coverage mismatch"):
            summarize_evaluation(root)

    def test_original_selection_requires_original_timestep_coverage(self):
        rows = []
        rows.extend(condition_rows("p1", "drive_a", 10, 250))
        rows.extend(condition_rows("p1", "drive_a", 11, 750))
        rows.extend(condition_rows("o1", "drive_old", 20, 250, cohort="original"))
        tmp, root = self.make_eval_dir(rows, ["p1"], timesteps=(250, 750), original_ids=["o1"])
        self.addCleanup(tmp.cleanup)
        with self.assertRaisesRegex(EvaluationSummaryError, "original selection coverage mismatch"):
            summarize_evaluation(root)

    def test_world_size_requires_reproduction_and_completion_rank_files(self):
        rows = condition_rows("p1", "drive_a", 10, 250)
        tmp, root = self.make_eval_dir(rows, ["p1"], world_size=2)
        self.addCleanup(tmp.cleanup)
        (root / "reproduction_rank1.json").unlink()
        with self.assertRaisesRegex(EvaluationSummaryError, "reproduction rank coverage mismatch"):
            summarize_evaluation(root)

        tmp2, root2 = self.make_eval_dir(rows, ["p1"], world_size=2)
        self.addCleanup(tmp2.cleanup)
        (root2 / "expanded_complete_rank1.json").unlink()
        (root2 / "original_complete_rank1.json").write_text(json.dumps({"passed": True}))
        with self.assertRaisesRegex(EvaluationSummaryError, "completion rank coverage mismatch"):
            summarize_evaluation(root2)

    def test_region_count_ranges_and_fraction_consistency_are_required(self):
        rows = condition_rows("p1", "drive_a", 10, 250)
        rows[0]["region_eps"]["valid_fraction"] = 0.24
        tmp, root = self.make_eval_dir(rows, ["p1"])
        self.addCleanup(tmp.cleanup)
        with self.assertRaisesRegex(EvaluationSummaryError, "valid_fraction does not equal"):
            summarize_evaluation(root)

        rows = condition_rows("p1", "drive_a", 10, 250)
        rows[0]["region_eps"]["valid_cells"] = 101
        tmp2, root2 = self.make_eval_dir(rows, ["p1"])
        self.addCleanup(tmp2.cleanup)
        with self.assertRaisesRegex(EvaluationSummaryError, "invalid valid_cells"):
            summarize_evaluation(root2)

    def test_pooled_bootstrap_tracks_drive_blocks_not_global_frame_blocks(self):
        rows = []
        for drive in ("drive_a", "drive_b"):
            for i, frame in enumerate((10, 120)):
                rows.extend(condition_rows(f"{drive}_{i}", drive, frame, 250))
        ids = ["drive_a_0", "drive_a_1", "drive_b_0", "drive_b_1"]
        tmp, root = self.make_eval_dir(rows, ids)
        self.addCleanup(tmp.cleanup)

        summary = summarize_evaluation(root, bootstrap_samples=10, seed=1)
        boot = summary["aggregates"]["expanded"]["250"]["pooled_micro"]["gains"]["all"][
            "disabled_minus_correct"
        ]["bootstrap"]
        self.assertEqual(boot["drive_count"], 2)
        self.assertEqual(boot["block_count"], 4)
        self.assertFalse(boot["degenerate"])

    def test_output_path_refuses_existing_files_without_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "summary.json").write_text("{}")
            with self.assertRaisesRegex(EvaluationSummaryError, "exists"):
                output_path(root, "summary.json", overwrite=False)
            self.assertEqual(output_path(root, "summary.json", overwrite=True), root / "summary.json")


if __name__ == "__main__":
    unittest.main()
