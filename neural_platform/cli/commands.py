"""
NeuralForge CLI
All commands accessible via the `neural` entry point.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional

import click
from rich.console import Console
from rich.table import Table

console = Console()


@click.group()
@click.version_option(version="0.4.2", prog_name="neural")
def cli():
    """
    NeuralForge — Build, train, and deploy neural networks from config.

    \b
    Quick start:
        neural init --model mlp --name my_experiment
        neural train --config runs/my_experiment/config.yaml
        neural serve --config runs/my_experiment/config.yaml
        neural dashboard
    """
    pass


# ---------------------------------------------------------------------------
# neural init
# ---------------------------------------------------------------------------

@cli.command()
@click.option("--model", "-m",
              type=click.Choice(["mlp", "cnn", "rnn", "transformer",
                                 "audio_cnn", "tcn", "tabular", "video_cnn"]),
              default="mlp", help="Model type to scaffold")
@click.option("--name", "-n", default="my_experiment", help="Experiment name")
@click.option("--output-dir", "-o", default="runs", help="Output directory root")
@click.option("--framework", "-f", type=click.Choice(["pytorch", "tensorflow", "jax"]),
              default="pytorch", help="Training framework")
@click.option("--force", is_flag=True, help="Overwrite existing config")
def init(model: str, name: str, output_dir: str, framework: str, force: bool):
    """Scaffold a new experiment config file."""
    from neural_platform.cli._templates import get_template_config
    import yaml

    run_dir = Path(output_dir) / name
    config_path = run_dir / "config.yaml"

    if config_path.exists() and not force:
        console.print(f"[yellow]Config already exists: {config_path}[/]")
        console.print("Use [bold]--force[/] to overwrite.")
        return

    run_dir.mkdir(parents=True, exist_ok=True)
    template = get_template_config(model, name, output_dir, framework)
    with open(config_path, "w") as f:
        yaml.dump(template, f, default_flow_style=False, allow_unicode=True, sort_keys=False)

    console.print(f"[green]✓ Config created:[/] {config_path}")
    console.print(f"\nEdit the config, then run:")
    console.print(f"  [bold cyan]neural train --config {config_path}[/]")


# ---------------------------------------------------------------------------
# neural deps
# ---------------------------------------------------------------------------

@cli.command()
@click.option("--model", "-m", default=None,
              help="Model type to scope the check (e.g. audio_cnn). Default: probe everything.")
@click.option("--source", "-s", default=None,
              help="Data source to scope the check (e.g. huggingface). Default: probe everything.")
@click.option("--json", "as_json", is_flag=True, help="Emit JSON instead of pretty output")
def deps(model: Optional[str], source: Optional[str], as_json: bool):
    """Audit installed Python packages for each modality / data source.

    Run with no arguments to see the full matrix:

        neural deps                # all model types + sources
        neural deps -m audio_cnn   # just the audio path
        neural deps --json         # for piping to jq
    """
    from neural_platform.core.deps import check_dependencies, check_all, format_report, install_command

    if model or source:
        # Scoped check
        report = check_dependencies(model or "mlp", source)
        if as_json:
            import json as _json
            click.echo(_json.dumps({
                "model": model, "source": source, "ok": report.ok,
                "missing_required": [d.package for d in report.missing_required],
                "missing_optional": [d.package for d in report.missing_optional],
                "statuses": [vars(s) for s in report.statuses],
            }, indent=2))
            return
        scope = []
        if model:  scope.append(f"model={model}")
        if source: scope.append(f"source={source}")
        console.print(f"[bold]Dependency check[/] ({', '.join(scope)})")
        console.print(format_report(report))
        cmd = install_command(report)
        if cmd:
            console.print()
            console.print(f"[yellow]To install missing packages:[/]\n  {cmd}")
        sys.exit(0 if report.ok else 2)

    # Full audit — show every modality
    all_reports = check_all()
    if as_json:
        import json as _json
        out = {}
        for key, rep in all_reports.items():
            out[key] = {
                "ok": rep.ok,
                "missing_required": [d.package for d in rep.missing_required],
                "missing_optional": [d.package for d in rep.missing_optional],
                "statuses": [vars(s) for s in rep.statuses],
            }
        click.echo(_json.dumps(out, indent=2))
        return

    from rich.table import Table
    table = Table(title="NeuralForge dependency matrix",
                  show_header=True, header_style="bold blue")
    table.add_column("Scope")
    table.add_column("Status")
    table.add_column("Missing required", style="red")
    table.add_column("Missing optional", style="yellow")
    for key, rep in all_reports.items():
        table.add_row(
            key,
            "[green]✓ ok[/]" if rep.ok else "[red]✗ missing[/]",
            ", ".join(d.package for d in rep.missing_required) or "—",
            ", ".join(d.package for d in rep.missing_optional) or "—",
        )
    console.print(table)

    # Aggregate install line
    all_missing_req: set = set()
    all_missing_opt: set = set()
    for rep in all_reports.values():
        all_missing_req.update(d.package for d in rep.missing_required)
        all_missing_opt.update(d.package for d in rep.missing_optional)
    if all_missing_req:
        console.print()
        console.print(f"[red]Required packages missing:[/]  {' '.join(sorted(all_missing_req))}")
        console.print(f"  pip install {' '.join(sorted(all_missing_req))}")
    if all_missing_opt:
        console.print()
        console.print(f"[yellow]Optional packages missing:[/] {' '.join(sorted(all_missing_opt))}")
        console.print(f"  pip install {' '.join(sorted(all_missing_opt))}")


# ---------------------------------------------------------------------------
# neural inspect
# ---------------------------------------------------------------------------

@cli.command()
@click.argument("dataset_name")
@click.option("--split", default=None, help="Which split to inspect (default: peek at the metadata only)")
def inspect(dataset_name: str, split: Optional[str]):
    """Inspect a HuggingFace dataset's columns/features without downloading the data.

    Use this to figure out what `data.text_column` / `data.label_column` to set,
    or to check whether a dataset is image vs text before plugging it into a config.
    """
    try:
        from datasets import load_dataset_builder
    except ImportError:
        console.print("[red]The `datasets` package is required.[/] Install with: pip install datasets")
        sys.exit(1)

    try:
        builder = load_dataset_builder(dataset_name)
    except Exception as exc:
        console.print(f"[red]Failed to fetch metadata for '{dataset_name}':[/] {exc}")
        sys.exit(1)

    info = builder.info
    console.print(f"\n[bold]Dataset:[/] {dataset_name}")
    if info.description:
        first_line = info.description.strip().split("\n")[0][:140]
        console.print(f"[dim]{first_line}[/]")

    # Available splits
    splits = list((info.splits or {}).keys())
    console.print(f"\n[bold]Splits:[/] {', '.join(splits) if splits else '(none discovered)'}")

    # Configurations
    configs = list(getattr(builder, "BUILDER_CONFIGS", []) or [])
    if configs:
        names = [getattr(c, "name", "?") for c in configs]
        console.print(f"[bold]Configurations:[/] {', '.join(names)}")

    # Features → table of columns by inferred kind
    features = getattr(info, "features", None)
    if features:
        from rich.table import Table
        table = Table(title="Columns", show_header=True, header_style="bold blue")
        table.add_column("Name")
        table.add_column("Type")
        table.add_column("Detected role", style="cyan")
        from neural_platform.core.validator import _features_summary
        summary = _features_summary(features)
        for col, feat in features.items():
            t = type(feat).__name__
            role = (
                "image"  if col in summary["image_columns"] else
                "label"  if col in summary["label_columns"] else
                "text"   if col in summary["text_columns"] else
                "other"
            )
            table.add_row(col, t, role)
        console.print()
        console.print(table)

        # Suggest a config snippet
        suggested_model = (
            "cnn" if summary["has_images"] and not summary["has_text"]
            else "transformer" if summary["has_text"]
            else "mlp"
        )
        text_col = (summary["text_columns"] or [None])[0]
        image_col = (summary["image_columns"] or [None])[0]
        label_col = (summary["label_columns"] or [None])[0]
        console.print()
        console.print("[bold]Suggested config snippet:[/]")
        snippet = ["model:",
                   f"  type: {suggested_model}",
                   "data:",
                   "  source: huggingface",
                   f"  dataset_name: {dataset_name}"]
        if text_col:
            snippet.append(f"  text_column: {text_col}")
        if image_col and suggested_model == "cnn":
            snippet.append(f"  # image column: {image_col} (auto-detected)")
        if label_col:
            snippet.append(f"  label_column: {label_col}")
        console.print("[dim]" + "\n".join(snippet) + "[/]")
    else:
        console.print("[yellow]No feature schema available — try loading the dataset directly.[/]")


# ---------------------------------------------------------------------------
# neural validate
# ---------------------------------------------------------------------------

@cli.command()
@click.option("--config", "-c", required=True, type=click.Path(exists=True), help="Path to config YAML/JSON")
@click.option("--strict", is_flag=True, help="Treat warnings as errors")
@click.option("--json", "as_json", is_flag=True, help="Emit JSON instead of pretty output")
def validate(config: str, strict: bool, as_json: bool):
    """Validate an experiment config without training."""
    from neural_platform.core.config import load_config
    from neural_platform.core.validator import validate_config

    try:
        cfg = load_config(config)
    except Exception as e:
        if as_json:
            import json as _json
            click.echo(_json.dumps({"ok": False, "schema_error": str(e)}))
        else:
            console.print(f"[red]Schema error:[/] {e}")
        sys.exit(2)

    report = validate_config(cfg)
    if as_json:
        import json as _json
        click.echo(_json.dumps(report.to_dict(), indent=2))
    else:
        for issue in report.issues:
            color = "red" if issue.severity == "error" else "yellow"
            console.print(f"[{color}]{issue.fmt()}[/]")
        if report.ok and not report.warnings:
            console.print(f"[green]✓ Config is valid:[/] {cfg.name} ({cfg.model.type.value})")
        elif report.ok:
            console.print(f"[green]✓ {len(report.warnings)} warning(s), no errors. Safe to train.[/]")
        else:
            console.print(f"[red]✗ {len(report.errors)} error(s). Fix before training.[/]")

    if not report.ok or (strict and report.warnings):
        sys.exit(2)


# ---------------------------------------------------------------------------
# neural train
# ---------------------------------------------------------------------------

@cli.command()
@click.option("--config", "-c", required=True, type=click.Path(exists=True), help="Path to config YAML/JSON")
@click.option("--override", "-O", multiple=True, help="Override config values: key=value (e.g. training.lr=0.001)")
@click.option("--db", default=None, help="Path to experiments database (default: <output_dir>/neuralforge.db)")
@click.option("--resume", "resume_path", type=click.Path(exists=True), default=None,
              help="Continue training from a checkpoint .pt — restores model weights, optimizer "
                   "state, scheduler state, and the epoch counter. Use this to extend a run.")
@click.option("--finetune", "finetune_path", type=click.Path(exists=True), default=None,
              help="Fine-tune from a checkpoint .pt — restores ONLY the model weights. Optimizer "
                   "and scheduler start fresh; the epoch counter resets. Mutually exclusive with --resume.")
@click.option("--resume-from-best", "resume_from_best", default=None,
              help="Shorthand: pass a run name (or path to its run_dir) and the trainer picks "
                   "<run>/checkpoints/checkpoint_best.pt. Combine with --resume or --finetune to "
                   "pick the mode (default: --resume).")
def train(config: str, override: tuple, db: Optional[str],
          resume_path: Optional[str], finetune_path: Optional[str],
          resume_from_best: Optional[str]):
    """Train a model from a config file.

    \b
    Resume vs. fine-tune:
      neural train -c cfg.yaml --resume runs/exp/checkpoints/checkpoint_best.pt
        Continues exactly where the run left off (optimizer/scheduler/epoch).

      neural train -c cfg.yaml --finetune runs/imdb/checkpoints/checkpoint_best.pt
        Loads only the weights; optimizer + scheduler start fresh.
        Use this when adapting a model to a new dataset or lowering the LR.
    """
    from neural_platform.core.config import load_config
    from neural_platform.core.trainer import Trainer
    from neural_platform.data.loader import build_dataloaders

    # Mutually-exclusive resume flags. We surface a clear error rather than
    # silently picking one — the two modes have very different semantics.
    if resume_path and finetune_path:
        console.print("[red]--resume and --finetune are mutually exclusive.[/] "
                      "Pick one: --resume preserves optimizer state, --finetune drops it.")
        sys.exit(2)

    # --resume-from-best resolves a run name → checkpoint path. Mode follows
    # whichever explicit flag was set; default to --resume if neither.
    if resume_from_best:
        if resume_path or finetune_path:
            console.print("[red]--resume-from-best can't be combined with an explicit "
                          "--resume / --finetune path.[/]")
            sys.exit(2)
        ckpt = _resolve_best_for_run(resume_from_best)
        if not ckpt:
            console.print(f"[red]No checkpoint_best.pt found for run '{resume_from_best}'.[/]")
            sys.exit(2)
        resume_path = ckpt   # default to full-state resume; users can pass --finetune explicitly

    resume_target = resume_path or finetune_path
    resume_mode = "weights_only" if finetune_path else "full"

    cfg = load_config(config)

    # Apply CLI overrides
    if override:
        cfg = _apply_overrides(cfg, override)

    console.print(f"[bold]Loaded config:[/] {config}")
    console.print(f"  Experiment:  {cfg.name}")
    console.print(f"  Model type:  {cfg.model.type.value}")
    console.print(f"  Framework:   {cfg.model.framework.value}")
    console.print(f"  Epochs:      {cfg.training.num_epochs}")
    console.print()

    # Pre-flight validation — fail fast on bad configs
    from neural_platform.core.validator import validate_config
    report = validate_config(cfg)
    for issue in report.warnings:
        console.print(f"[yellow]⚠[/] {issue.field}: {issue.message}")
        if issue.hint:
            console.print(f"    [dim]→ {issue.hint}[/]")
    if not report.ok:
        console.print()
        for issue in report.errors:
            console.print(f"[red]✗[/] {issue.field}: {issue.message}")
            if issue.hint:
                console.print(f"    [dim]→ {issue.hint}[/]")
        console.print()
        console.print(f"[red]Config has {len(report.errors)} error(s). Fix and re-run, "
                       "or `neural validate` to see details.[/]")
        sys.exit(2)

    # Build data loaders
    try:
        train_loader, val_loader, _ = build_dataloaders(
            cfg.data, cfg.training, cfg.model.type.value, model_cfg=cfg.model,
        )
    except Exception as e:
        console.print(f"[red]Error loading data:[/] {e}")
        sys.exit(1)

    # Try to discover class names from the data and stash them on the config
    # for downstream use (checkpoint, inference server /info).
    try:
        from neural_platform.data.loader import get_class_names
        names = get_class_names(train_loader) or get_class_names(val_loader)
        if names:
            cfg._class_names = names  # type: ignore[attr-defined]
    except Exception:
        pass

    n_train = len(train_loader.dataset)
    n_val = len(val_loader.dataset) if val_loader else 0
    console.print(f"  Train samples: {n_train:,}")
    console.print(f"  Val samples:   {n_val:,}")
    console.print()

    if resume_target:
        action = "Resuming" if resume_mode == "full" else "Fine-tuning"
        console.print(f"[bold]{action} from:[/] {resume_target}")

    trainer = Trainer(cfg, db_path=db)
    try:
        result = trainer.fit(
            train_loader, val_loader,
            resume_from=resume_target,
            resume_mode=resume_mode,
        )
    except Exception as e:
        console.print(f"[red]Training failed:[/] {e}")
        raise

    if result["best_checkpoint"]:
        console.print(f"\n[green]Best checkpoint:[/] {result['best_checkpoint']}")
        console.print(f"[green]To serve:[/] neural serve --config {config}")


# ---------------------------------------------------------------------------
# neural evaluate
# ---------------------------------------------------------------------------

@cli.command()
@click.option("--config", "-c", required=True, type=click.Path(exists=True))
@click.option("--checkpoint", "-k", default=None, help="Path to checkpoint (default: best in run dir)")
@click.option("--split", "-s", type=click.Choice(["val", "test", "train"]), default="val")
def evaluate(config: str, checkpoint: Optional[str], split: str):
    """Evaluate a trained model on val/test data."""
    from neural_platform.core.config import load_config
    from neural_platform.core.evaluator import Evaluator
    from neural_platform.data.loader import build_dataloaders
    from neural_platform.frameworks.factory import get_adapter

    cfg = load_config(config)
    adapter = get_adapter(cfg)
    _, val_loader, test_loader = build_dataloaders(cfg.data, cfg.training, cfg.model.type.value)

    loader = {"val": val_loader, "test": test_loader, "train": None}[split]
    if loader is None:
        console.print(f"[red]No {split} data available.[/]")
        sys.exit(1)

    # Resolve checkpoint
    if not checkpoint:
        ckpt_dir = cfg.checkpoint_dir
        best = ckpt_dir / "checkpoint_best.pt"
        if best.exists():
            checkpoint = str(best)
        else:
            pts = sorted(ckpt_dir.glob("*.pt"))
            if pts:
                checkpoint = str(pts[-1])
            else:
                console.print("[red]No checkpoint found. Run `neural train` first.[/]")
                sys.exit(1)

    console.print(f"[bold]Loading checkpoint:[/] {checkpoint}")
    model, _ = adapter.load_checkpoint(checkpoint)
    loss_fn = adapter.build_loss()
    evaluator = Evaluator(adapter, loss_fn)

    metrics = evaluator.evaluate(model, loader, phase=split)
    console.print(f"\n[bold]{split.capitalize()} Metrics:[/]")
    table = Table(show_header=True)
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    for k, v in metrics.items():
        table.add_row(k, f"{v:.6f}")
    console.print(table)


# ---------------------------------------------------------------------------
# neural serve
# ---------------------------------------------------------------------------

@cli.command()
@click.option("--config", "-c", required=True, type=click.Path(exists=True))
@click.option("--checkpoint", "-k", default=None, help="Checkpoint to serve")
@click.option("--host", default=None, help="Override bind host")
@click.option("--port", "-p", default=None, type=int, help="Override bind port")
@click.option("--reload", is_flag=True, help="Enable hot reload (dev)")
@click.option("--no-auth", is_flag=True,
              help="Disable bearer-token auth. NOT recommended on shared networks.")
@click.option("--no-checkpoint", is_flag=True,
              help="Serve the model without loading a NeuralForge checkpoint. "
                   "Only valid for hf_pipeline models — their weights come from "
                   "HuggingFace at startup. Use this for zero-shot inference on a "
                   "published HF model that you haven't fine-tuned locally.")
def serve(config: str, checkpoint: Optional[str], host: Optional[str], port: Optional[int],
          reload: bool, no_auth: bool, no_checkpoint: bool):
    """Launch the REST inference server for a trained model.

    By default the server requires `Authorization: Bearer <token>` on every
    endpoint except /health. If `NEURAL_INFERENCE_TOKEN` is set the server
    uses that; otherwise it generates a one-shot token at startup and prints
    it to stderr. Pass `--no-auth` to disable auth entirely.

    Pass `--no-checkpoint` for HuggingFace pipeline models when you want to
    serve the published weights as-is, with no local training run behind them.
    """
    import os, uvicorn
    from neural_platform.core.config import load_config
    from neural_platform.deploy.server import create_inference_app

    cfg = load_config(config)
    deploy_cfg = cfg.deploy
    bind_host = host or deploy_cfg.host
    bind_port = port or deploy_cfg.port

    # `--no-checkpoint` short-circuits the usual checkpoint resolution. The
    # combination `--checkpoint X --no-checkpoint` is contradictory — refuse it.
    if no_checkpoint and checkpoint:
        console.print("[red]--checkpoint and --no-checkpoint are mutually exclusive.[/] "
                      "Drop one.")
        sys.exit(2)

    if no_checkpoint:
        if cfg.model.type.value != "hf_pipeline":
            console.print(
                f"[red]--no-checkpoint is only valid for hf_pipeline models, "
                f"got '{cfg.model.type.value}'.[/] Other model types need a "
                "NeuralForge checkpoint with trained weights."
            )
            sys.exit(2)
        checkpoint = None
    else:
        # Resolve checkpoint
        if not checkpoint:
            ckpt_dir = cfg.checkpoint_dir
            best = ckpt_dir / "checkpoint_best.pt"
            checkpoint = str(best) if best.exists() else None

        if not checkpoint or not Path(checkpoint).exists():
            console.print("[red]No checkpoint found. Run `neural train` first, "
                          "or pass --no-checkpoint for hf_pipeline models.[/]")
            sys.exit(1)

    if no_auth:
        os.environ["NEURAL_INFERENCE_AUTH"] = "off"

    console.print(f"[bold green]NeuralForge Inference Server[/]")
    console.print(f"  Model:      {cfg.model.type.value} ({cfg.model.name})")
    if checkpoint:
        console.print(f"  Checkpoint: {checkpoint}")
    else:
        pretrained = (cfg.model.hf_pipeline.pretrained
                      if cfg.model.hf_pipeline else "(unknown)")
        console.print(f"  Checkpoint: [yellow](none — serving HF weights for {pretrained})[/]")
    console.print(f"  Listening:  http://{bind_host}:{bind_port}")
    console.print(f"  Docs:       http://{bind_host}:{bind_port}/docs")
    auth_note = ("[red]disabled (--no-auth)[/]" if no_auth
                 else "[green]bearer token (printed below if auto-generated)[/]")
    console.print(f"  Auth:       {auth_note}\n")
    if bind_host in ("0.0.0.0", "::") and no_auth:
        console.print(
            "[yellow]⚠ Binding 0.0.0.0 with --no-auth makes the inference server "
            "reachable from any machine on this network without authentication. "
            "Bind to 127.0.0.1 or drop --no-auth.[/]"
        )

    app = create_inference_app(cfg, checkpoint)
    uvicorn.run(app, host=bind_host, port=bind_port, reload=reload)


# ---------------------------------------------------------------------------
# neural dashboard
# ---------------------------------------------------------------------------

@cli.command()
@click.option("--output-dir", "-o", default="runs", help="Experiments output directory")
@click.option("--host", default="127.0.0.1", help="Bind host")
@click.option("--port", "-p", default=7860, type=int, help="Bind port")
def dashboard(output_dir: str, host: str, port: int):
    """Launch the NeuralForge web dashboard."""
    import uvicorn
    from neural_platform.web.app import create_dashboard_app

    app = create_dashboard_app(output_dir)
    console.print(f"[bold green]NeuralForge Dashboard[/]")
    console.print(f"  Open: http://{host}:{port}\n")
    uvicorn.run(app, host=host, port=port, reload=False)


# ---------------------------------------------------------------------------
# neural list
# ---------------------------------------------------------------------------

@cli.command("list")
@click.option("--output-dir", "-o", default="runs", help="Output dir to scan")
def list_experiments(output_dir: str):
    """List all tracked experiments."""
    from neural_platform.core.experiment import ExperimentTracker
    db_path = Path(output_dir) / "neuralforge.db"
    if not db_path.exists():
        console.print("[yellow]No experiments found. Run `neural train` first.[/]")
        return

    tracker = ExperimentTracker(db_path)
    experiments = tracker.list_experiments()
    if not experiments:
        console.print("[yellow]No experiments in database.[/]")
        return

    table = Table(title="Experiments", show_header=True, header_style="bold blue")
    table.add_column("ID", style="dim")
    table.add_column("Name", style="cyan")
    table.add_column("Status")
    table.add_column("Created")
    table.add_column("Tags")
    for exp in experiments:
        tags = ", ".join(json.loads(exp.get("tags", "[]")))
        status_color = "green" if exp["status"] == "completed" else "yellow"
        table.add_row(
            str(exp["id"]),
            exp["name"],
            f"[{status_color}]{exp['status']}[/]",
            exp["created_at"][:19],
            tags,
        )
    console.print(table)


# ---------------------------------------------------------------------------
# neural status
# ---------------------------------------------------------------------------

@cli.command()
@click.argument("experiment_id", type=int)
@click.option("--output-dir", "-o", default="runs")
def status(experiment_id: int, output_dir: str):
    """Show detailed status of an experiment."""
    from neural_platform.core.experiment import ExperimentTracker
    db_path = Path(output_dir) / "neuralforge.db"
    tracker = ExperimentTracker(db_path)

    exp = tracker.get_experiment(experiment_id)
    if not exp:
        console.print(f"[red]Experiment {experiment_id} not found.[/]")
        return

    console.print(f"\n[bold]{exp['name']}[/] (ID: {exp['id']})")
    console.print(f"  Status:  {exp['status']}")
    console.print(f"  Created: {exp['created_at']}")

    runs = tracker.list_runs(experiment_id)
    if runs:
        table = Table(title="Runs", show_header=True, header_style="bold")
        table.add_column("Run #")
        table.add_column("Framework")
        table.add_column("Status")
        table.add_column("Best Val Loss")
        table.add_column("Best Epoch")
        table.add_column("Duration")
        for run in runs:
            dur = f"{run['duration_secs']:.0f}s" if run["duration_secs"] else "—"
            table.add_row(
                str(run["run_number"]),
                run["framework"] or "—",
                run["status"],
                f"{run['best_val_loss']:.6f}" if run["best_val_loss"] is not None else "—",
                str(run["best_epoch"] or "—"),
                dur,
            )
        console.print(table)


# ---------------------------------------------------------------------------
# neural export
# ---------------------------------------------------------------------------

@cli.command()
@click.option("--config", "-c", required=True, type=click.Path(exists=True))
@click.option("--checkpoint", "-k", default=None)
@click.option("--format", "-f", "fmt", type=click.Choice(["onnx", "torchscript"]), default="onnx")
@click.option("--output", "-o", default=None, help="Output file path")
def export(config: str, checkpoint: Optional[str], fmt: str, output: Optional[str]):
    """Export a trained model to ONNX or TorchScript format."""
    import torch
    from neural_platform.core.config import load_config
    from neural_platform.frameworks.factory import get_adapter

    cfg = load_config(config)
    adapter = get_adapter(cfg)

    if not checkpoint:
        best = cfg.checkpoint_dir / "checkpoint_best.pt"
        checkpoint = str(best) if best.exists() else None
    if not checkpoint or not Path(checkpoint).exists():
        console.print("[red]No checkpoint found.[/]")
        sys.exit(1)

    model, _ = adapter.load_checkpoint(checkpoint)
    model.eval()

    out_path = output or str(cfg.run_dir / f"model.{fmt}")

    if fmt == "torchscript":
        scripted = torch.jit.script(model)
        scripted.save(out_path)
    else:
        # ONNX — create a dummy input based on model type
        mtype = cfg.model.type.value
        arch = cfg.model.get_arch_config()
        if mtype == "mlp":
            dummy = torch.randn(1, arch.input_size)
        elif mtype == "cnn":
            dummy = torch.randn(1, arch.input_channels, arch.input_height, arch.input_width)
        elif mtype == "rnn":
            dummy = torch.randn(1, 32, arch.input_size)
        elif mtype == "transformer":
            dummy = torch.randint(0, arch.vocab_size, (1, 32))
        else:
            console.print(f"[red]ONNX export not implemented for model type: {mtype}[/]")
            sys.exit(1)

        torch.onnx.export(
            model, dummy, out_path,
            opset_version=17,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
        )

    console.print(f"[green]✓ Exported to {out_path}[/]")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolve_best_for_run(name_or_path: str) -> Optional[str]:
    """Map a run name (e.g. ``imdb_lstm``) or run-dir path to its
    ``checkpoints/checkpoint_best.pt`` if one exists.

    Search order:
      1. Treat the input as a path; if it points at a .pt file directly,
         return it.
      2. If it's a directory, look for ``checkpoints/checkpoint_best.pt`` and
         then for the newest ``checkpoints/*.pt``.
      3. Treat it as a run name under ``runs/<name>/checkpoints/...``.

    Returns a string path if found, else None. The CLI surfaces a clear
    error in the latter case.
    """
    candidate = Path(name_or_path)

    # 1) Direct .pt path
    if candidate.suffix == ".pt" and candidate.exists():
        return str(candidate.resolve())

    # 2) Directory
    if candidate.is_dir():
        best = candidate / "checkpoints" / "checkpoint_best.pt"
        if best.exists():
            return str(best.resolve())
        ckpts = sorted((candidate / "checkpoints").glob("*.pt"),
                       key=lambda p: p.stat().st_mtime, reverse=True) \
                if (candidate / "checkpoints").is_dir() else []
        if ckpts:
            return str(ckpts[0].resolve())
        return None

    # 3) Bare run name → `runs/<name>/checkpoints/checkpoint_best.pt`
    bare = Path("runs") / name_or_path / "checkpoints" / "checkpoint_best.pt"
    if bare.exists():
        return str(bare.resolve())
    bare_dir = Path("runs") / name_or_path / "checkpoints"
    if bare_dir.is_dir():
        ckpts = sorted(bare_dir.glob("*.pt"),
                       key=lambda p: p.stat().st_mtime, reverse=True)
        if ckpts:
            return str(ckpts[0].resolve())
    return None


def _apply_overrides(cfg, overrides: tuple):
    """Apply --override key=value strings to a config."""
    import yaml
    data = cfg.model_dump()
    for ov in overrides:
        if "=" not in ov:
            console.print(f"[yellow]Ignoring malformed override (no '='): {ov}[/]")
            continue
        key, _, val = ov.partition("=")
        parts = key.strip().split(".")
        node = data
        for part in parts[:-1]:
            if part not in node:
                console.print(f"[yellow]Override key not found: {key}[/]")
                break
            node = node[part]
        else:
            # Try to coerce the value
            try:
                node[parts[-1]] = yaml.safe_load(val)
            except Exception:
                node[parts[-1]] = val
    from neural_platform.core.config import ExperimentConfig
    return ExperimentConfig.model_validate(data)
