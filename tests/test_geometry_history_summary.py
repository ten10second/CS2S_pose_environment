import json
import tempfile
import unittest
from pathlib import Path

from tools.summarize_geometry_history_run import summarize_run


def write_jsonl(path, rows):
    with Path(path).open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def probe_rows(rank, steps=(0, 5), disabled_loss=1.0, final_correct_delta=0.2):
    rows = []
    for step in steps:
        for split in ("train", "val"):
            for timestep in (250, 750):
                losses = {
                    "disabled": disabled_loss,
                    "correct": disabled_loss if step == 0 else disabled_loss - final_correct_delta,
                    "wrong_geometry": disabled_loss if step == 0 else disabled_loss - 0.05,
                    "wrong_history": disabled_loss if step == 0 else disabled_loss - 0.1,
                }
                for condition, loss in losses.items():
                    rows.append(
                        {
                            "rank": rank,
                            "step": step,
                            "split": split,
                            "t": timestep,
                            "condition": condition,
                            "loss_total": loss,
                            "pair_id": f"{split}_{rank}",
                        }
                    )
    return rows


def metric_rows(rank, steps=range(1, 7)):
    rows = []
    for step in steps:
        history = step != 2
        value = 0.0 if step == 2 else 0.1 + rank
        rows.append(
            {
                "rank": rank,
                "step": step,
                "loss": 0.5,
                "encoder_grad_l2": value,
                "cond_query_grad_l2": value,
                "out_grad_l2": value,
                "history": history,
                "satellite_dropped": step == 3,
            }
        )
    return rows


class GeometryHistorySummaryTests(unittest.TestCase):
    def make_run(self, world_size=2, steps=6, checkpoint=True):
        tmp = tempfile.TemporaryDirectory()
        root = Path(tmp.name)
        (root / "run.json").write_text(
            json.dumps(
                {
                    "mode": "geometry_history_v1",
                    "world_size": world_size,
                    "train_pairs": 100,
                    "val_pairs": 4,
                    "val_drives": ["drive_val"],
                    "args": {"steps": steps},
                }
            )
        )
        for rank in range(world_size):
            write_jsonl(root / f"metrics_rank{rank}.jsonl", metric_rows(rank, range(1, steps + 1)))
            write_jsonl(root / f"probes_rank{rank}.jsonl", probe_rows(rank, steps=(0, steps)))
        if checkpoint:
            (root / f"geometry_history_step_{steps}.pt").write_bytes(b"checkpoint")
        return tmp, root

    def test_complete_run_smoke_passes_and_aggregates_probe_benefits(self):
        tmp, root = self.make_run()
        self.addCleanup(tmp.cleanup)
        summary = summarize_run(root)
        self.assertEqual(summary["status"], "complete")
        self.assertTrue(summary["smoke"]["complete_steps_all_world_ranks"])
        self.assertTrue(summary["smoke"]["checkpoint_exists_at_final_step"])
        self.assertTrue(summary["smoke"]["step0_all_conditions_identical"]["pass"])
        self.assertTrue(summary["smoke"]["disabled_probe_constant"]["pass"])
        self.assertTrue(summary["smoke"]["finite_loss_all_ranks"])
        self.assertTrue(summary["smoke"]["finite_grad_metrics_all_ranks"])
        self.assertTrue(summary["smoke"]["enabled_history_gradients_nonzero_after_step5"]["pass"])
        self.assertTrue(summary["smoke"]["no_history_step2_zero_gradients_all_ranks"]["pass"])
        self.assertTrue(summary["smoke"]["cfg_step3_logged_true_all_ranks"]["pass"])
        paired = summary["fixed_probes"]["paired"]
        final_train = [r for r in paired if r["split"] == "train" and r["step"] == 6]
        self.assertEqual(len(final_train), 2)
        self.assertTrue(all(r["benefit_disabled_minus_correct"] > 0 for r in final_train))
        self.assertTrue(all(r["benefit_disabled_minus_correct_count"] == 2 for r in final_train))
        positives = summary["fixed_probes"]["positive_counts"]
        item = [
            r for r in positives
            if r["split"] == "val"
            and r["step"] == 6
            and r["metric"] == "benefit_wrong_geometry_minus_correct"
        ][0]
        self.assertEqual(item["positive_count"], item["count"])
        self.assertIn("not a full benchmark", summary["metadata"]["note"])

    def test_partial_run_is_incomplete_not_failed(self):
        tmp, root = self.make_run(steps=10, checkpoint=False)
        self.addCleanup(tmp.cleanup)
        write_jsonl(root / "metrics_rank1.jsonl", metric_rows(1, range(1, 5)))
        summary = summarize_run(root)
        self.assertEqual(summary["status"], "incomplete")
        self.assertFalse(summary["smoke"]["complete_steps_all_world_ranks"])
        self.assertFalse(summary["smoke"]["checkpoint_exists_at_final_step"])
        self.assertIn(5, summary["training"]["1"]["missing_steps"])

    def test_step0_mismatch_is_reported(self):
        tmp, root = self.make_run()
        self.addCleanup(tmp.cleanup)
        rows = probe_rows(0, steps=(0, 6))
        rows[1]["loss_total"] = 1.1
        write_jsonl(root / "probes_rank0.jsonl", rows)
        summary = summarize_run(root)
        self.assertFalse(summary["smoke"]["step0_all_conditions_identical"]["pass"])
        self.assertTrue(summary["smoke"]["step0_all_conditions_identical"]["failures"])

    def test_disabled_drift_is_reported(self):
        tmp, root = self.make_run()
        self.addCleanup(tmp.cleanup)
        rows = probe_rows(0, steps=(0, 6))
        for row in rows:
            if row["step"] == 6 and row["condition"] == "disabled":
                row["loss_total"] = 1.01
                break
        write_jsonl(root / "probes_rank0.jsonl", rows)
        summary = summarize_run(root)
        self.assertFalse(summary["smoke"]["disabled_probe_constant"]["pass"])
        self.assertTrue(summary["smoke"]["disabled_probe_constant"]["failures"])

    def test_positive_counts_are_per_rank_pair_not_global_mean_sign(self):
        tmp, root = self.make_run(world_size=2, steps=6)
        self.addCleanup(tmp.cleanup)
        write_jsonl(root / "probes_rank0.jsonl", probe_rows(0, steps=(0, 6), final_correct_delta=0.2))
        write_jsonl(root / "probes_rank1.jsonl", probe_rows(1, steps=(0, 6), final_correct_delta=-0.2))
        summary = summarize_run(root)
        positives = summary["fixed_probes"]["positive_counts"]
        item = [
            r for r in positives
            if r["split"] == "train"
            and r["step"] == 6
            and r["t"] == 250
            and r["metric"] == "benefit_disabled_minus_correct"
        ][0]
        self.assertEqual(item["count"], 2)
        self.assertEqual(item["positive_count"], 1)

    def test_missing_intermediate_step_makes_run_incomplete_even_with_final_checkpoint(self):
        tmp, root = self.make_run(steps=6)
        self.addCleanup(tmp.cleanup)
        write_jsonl(root / "metrics_rank0.jsonl", metric_rows(0, [1, 2, 3, 5, 6]))
        summary = summarize_run(root)
        self.assertEqual(summary["status"], "incomplete")
        self.assertFalse(summary["training"]["0"]["complete_final_step"])
        self.assertEqual(summary["training"]["0"]["missing_steps"], [4])

    def test_resume_marks_fresh_smoke_checks_not_applicable(self):
        tmp, root = self.make_run(steps=8)
        self.addCleanup(tmp.cleanup)
        meta = json.loads((root / "run.json").read_text())
        meta["start_step"] = 4
        (root / "run.json").write_text(json.dumps(meta))
        for rank in range(2):
            write_jsonl(root / f"metrics_rank{rank}.jsonl", metric_rows(rank, range(5, 9)))
        (root / "geometry_history_step_8.pt").write_bytes(b"checkpoint")
        summary = summarize_run(root)
        self.assertEqual(summary["smoke"]["step0_all_conditions_identical"]["status"], "not_applicable")
        self.assertEqual(summary["smoke"]["no_history_step2_zero_gradients_all_ranks"]["status"], "not_applicable")
        self.assertEqual(summary["smoke"]["cfg_step3_logged_true_all_ranks"]["status"], "not_applicable")


if __name__ == "__main__":
    unittest.main()
