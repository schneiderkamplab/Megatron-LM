"""Build (or update) the wandb report that compares downstream accuracy, loss and
training time across the runs of each experiment group.

    python wandb_dashboard.py                      # backfill summaries, create a report
    python wandb_dashboard.py --report-url URL     # ... and update that report instead

First fills in the run summary values the plots use, for runs that lack them:
train_hours (summed training step time, without evaluation and startup),
final_train_loss (mean of the last 10 logged train/loss values) and
final_val_loss (from the history, for runs that crashed before writing it).
Downstream scores (eval/*) come from collect_evals.py.
"""

import argparse

import wandb
import wandb_workspaces.expr as expr
import wandb_workspaces.reports.v2 as wr

PROJECT = "kuchenpferd-org/detonation-megatron"

# (group, heading, description, task columns); the newest experiments first.
BASE_TASKS = ["arc_e", "arc_c", "hswag", "piqa", "wino", "sciq", "boolq", "obqa", "lambada"]
SFT_TASKS = BASE_TASKS + ["mmlu", "gsm8k", "ifeval"]
SECTIONS = [
    ("olmo2-7b-ablation-2n-20261124", "Sign step and optimizer ablation (7B, 2 nodes, 3k steps)",
     "Pretraining from scratch, synchronous replication only. -nosign: AdamW on the "
     "replicated gradient itself; -sgd: plain SGD on the sign (the reference DeMo update). "
     "At 3k steps downstream scores are close to chance, so the loss plots carry the signal.",
     BASE_TASKS),
    ("olmo2-7b-socket-8n-20261112", "Pretraining on a throttled network (7B, 8 nodes, TCP)",
     "Pretraining from scratch for 10k steps with inter-node traffic over TCP sockets. "
     "full-async and demo-async have no downstream scores (they crashed while saving).",
     BASE_TASKS),
    ("olmo2-7b-pretrain-2n-20261112", "Pretraining (7B, 2 nodes)",
     "Pretraining from scratch for 10k steps on the fast interconnect.", BASE_TASKS),
    ("olmo2-7b-sft-2n-20261111", "SFT (7B, 2 nodes)",
     "OLMo-2 7B fine-tuned on the Tulu 3 SFT mixture for 2k steps. For reference: base model "
     "MC avg 68.2, IFEval 18.8; AllenAI's OLMo-2 7B SFT MC avg 71.8, IFEval 61.6.",
     SFT_TASKS),
]
# Runs kept out of the plots: superseded variants and incomplete attempts.
EXCLUDED_TAGS = ["adaptive-v1", "partial", "superseded", "eval-only"]

# One hue per replicator; sync dark, async light, ablation variants in between.
HUES = {
    "none": ("#4d4d4d", "#4d4d4d", "#4d4d4d", "#4d4d4d"),
    "full": ("#1f5fbf", "#86b4ec", "#3f8fd9", "#0b2f66"),
    "demo": ("#d46a00", "#f5b766", "#ec9a3a", "#7a3d00"),
    "random": ("#b0175a", "#ef8fb5", "#d9548c", "#5e0b30"),
}


def run_color(name: str) -> str:
    sync, async_, nosign, sgd = HUES.get(name.split("-")[0], HUES["none"])
    if "-sgd" in name:
        return sgd
    if "nosign" in name:
        return nosign
    if "async" in name or "delayed" in name:
        return async_
    return sync


def backfill(api, group: str) -> list:
    """Fill in the summary values the plots use; return the group's runs."""
    runs = list(api.runs(PROJECT, filters={"group": group}))
    for run in runs:
        if run.state == "running":
            continue
        updates = {}
        if run.summary.get("train_hours") is None:
            interval = run.config.get("log_interval", 10)
            times = [row["perf/step_time_sec"] for row in
                     run.scan_history(keys=["perf/step_time_sec"]) if row["perf/step_time_sec"]]
            if times:
                updates["train_hours"] = sum(times) * interval / 3600
        if run.summary.get("final_train_loss") is None:
            losses = [row["train/loss"] for row in run.scan_history(keys=["train/loss"])
                      if row["train/loss"] is not None]
            if losses:
                updates["final_train_loss"] = sum(losses[-10:]) / len(losses[-10:])
        if run.summary.get("final_val_loss") is None:
            vals = [row["val/loss"] for row in run.scan_history(keys=["val/loss"])
                    if row["val/loss"] is not None]
            if vals:
                updates["final_val_loss"] = vals[-1]
        if updates:
            for key, value in updates.items():
                run.summary[key] = value
            run.summary.update()
            print(f"{group}/{run.name}: {updates}")
    return runs


def section_blocks(group, heading, description, tasks, runs):
    entity, project = PROJECT.split("/")
    runset = wr.Runset(
        entity=entity, project=project, name=heading,
        filters=[expr.Metric("Group") == group, expr.Tags().notin(EXCLUDED_TAGS)],
    )
    colors = {run.id: run_color(run.name) for run in runs}
    mc = wr.SummaryMetric("eval/mc_avg")
    panels = [
        wr.ScatterPlot(title="Downstream accuracy vs training time (top left is better)",
                       x=wr.SummaryMetric("train_hours"), y=mc),
        wr.ScatterPlot(title="Final validation loss vs training time (bottom left is better)",
                       x=wr.SummaryMetric("train_hours"), y=wr.SummaryMetric("final_val_loss")),
        wr.ScatterPlot(title="Downstream accuracy vs final validation loss",
                       x=wr.SummaryMetric("final_val_loss"), y=mc),
        wr.BarPlot(title="Downstream accuracy (MC avg)", metrics=[mc], orientation="h"),
        wr.ParallelCoordinatesPlot(
            title="Accuracy per task",
            columns=[wr.ParallelCoordinatesPlotColumn(metric=wr.SummaryMetric(f"eval/{t}"),
                                                      display_name=t)
                     for t in ["mc_avg"] + tasks],
        ),
        wr.LinePlot(title="Validation loss", x="Step", y=["val/loss"]),
    ]
    return [
        wr.H2(text=heading),
        wr.P(text=f"{description} wandb group: {group}."),
        wr.PanelGrid(runsets=[runset], panels=panels, custom_run_colors=colors),
    ]


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--report-url", default=None, help="update this report instead of creating one")
    args = p.parse_args()

    api = wandb.Api()
    blocks = [wr.P(text=(
        "One point per run. Hover a point for the run name. Colours: grey none, blue full, "
        "orange demo, pink random; dark = sync, light = async, mid = no sign step, "
        "darkest = SGD. MC avg is the mean accuracy over the 8 multiple-choice tasks, each "
        "on its first 500 examples, so differences under 2-3 points are noise. Training "
        "time is the summed step time, without evaluation and startup."))]
    for group, heading, description, tasks in SECTIONS:
        blocks += section_blocks(group, heading, description, tasks, backfill(api, group))

    entity, project = PROJECT.split("/")
    if args.report_url:
        report = wr.Report.from_url(args.report_url)
        report.blocks = blocks
    else:
        report = wr.Report(entity=entity, project=project, title="DeToNATION 7B overview",
                           description="Downstream accuracy, loss and training time per run.",
                           blocks=blocks, width="fluid")
    report.save()
    print(report.url)


if __name__ == "__main__":
    main()
