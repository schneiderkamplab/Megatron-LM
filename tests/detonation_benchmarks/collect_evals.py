"""Tabulate the quick downstream evaluations of a sweep and attach them to wandb.

Reads every lm-evaluation-harness results file under <ckpt_root>/<run>/eval/
(written by eval_lm.sbatch), prints one row per run, and, with --wandb-group,
stores the scores as eval/<task> in the summary of the wandb run of the same
name in that group.

    python collect_evals.py /scratch/.../checkpoints/<group> [--wandb-group <group>]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

# Headline metric per task: accuracy for multiple choice (length-normalized
# where the task defines it), exact match for GSM8K, prompt-level strict
# accuracy for IFEval.
METRICS = {
    "arc_easy": "acc_norm", "arc_challenge": "acc_norm", "hellaswag": "acc_norm",
    "piqa": "acc_norm", "winogrande": "acc", "sciq": "acc_norm", "boolq": "acc",
    "openbookqa": "acc_norm", "lambada_openai": "acc", "mmlu": "acc",
    "gsm8k_cot": "exact_match,flexible-extract", "ifeval": "prompt_level_strict_acc",
}
SHORT = {"arc_easy": "arc_e", "arc_challenge": "arc_c", "hellaswag": "hswag",
         "winogrande": "wino", "openbookqa": "obqa", "lambada_openai": "lambada",
         "gsm8k_cot": "gsm8k"}
MC_TASKS = ["arc_easy", "arc_challenge", "hellaswag", "piqa", "winogrande", "sciq",
            "boolq", "openbookqa"]


def load_scores(eval_dir: Path) -> dict[str, float]:
    scores = {}
    for path in sorted(eval_dir.glob("**/results_*.json")):  # later files win
        results = json.loads(path.read_text())["results"]
        for task, metric in METRICS.items():
            entry = results.get(task, {})
            value = entry.get(f"{metric},none", entry.get(metric))
            if value is not None:
                scores[task] = 100.0 * value
    mc = [scores[t] for t in MC_TASKS if t in scores]
    if len(mc) == len(MC_TASKS):
        scores["mc_avg"] = sum(mc) / len(mc)
    return scores


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("ckpt_root", type=Path)
    p.add_argument("--wandb-group", default=None)
    p.add_argument("--wandb-path", default="kuchenpferd-org/detonation-megatron")
    args = p.parse_args()

    rows = {run.name: load_scores(run / "eval")
            for run in sorted(args.ckpt_root.iterdir()) if (run / "eval").is_dir()}
    tasks = [t for t in ["mc_avg", *METRICS] if any(t in s for s in rows.values())]
    print(f"{'run':18}" + "".join(f"{SHORT.get(t, t):>9}" for t in tasks))
    for name, scores in rows.items():
        print(f"{name:18}" + "".join(
            f"{scores[t]:9.1f}" if t in scores else f"{'-':>9}" for t in tasks))

    if args.wandb_group:
        import wandb

        runs = {r.name: r for r in wandb.Api().runs(
            args.wandb_path, filters={"group": args.wandb_group})}
        for name, scores in rows.items():
            if name in runs and scores:
                for task, value in scores.items():
                    runs[name].summary[f"eval/{SHORT.get(task, task)}"] = value
                runs[name].summary.update()
                print(f"wandb: attached {len(scores)} scores to {name}")


if __name__ == "__main__":
    main()
