# NeuralForge

A multi-framework platform for **building**, **training**, and **serving** neural networks from a config file or a HuggingFace model id — driven by a single `neural` CLI and a real-time web dashboard.

NeuralForge is opinionated about the boring parts (config schema, checkpoint format, experiment tracking, REST inference) so you can focus on the model and the data.

```bash
# 1. Scaffold a config
neural init --model cnn --name cifar10_demo

# 2. Train
neural train -c runs/cifar10_demo/config.yaml

# 3. Serve the trained model on a REST endpoint
neural serve -c runs/cifar10_demo/config.yaml

# 4. Or skip training and serve a HuggingFace model directly
neural serve -c runs/whisper.yaml --no-checkpoint
```

---

## Features

- **Four training surfaces** — Python API, CLI, web dashboard, and a managed-server lifecycle for Predict.
- **Nine model families** — `mlp`, `cnn`, `rnn`, `transformer`, `audio_cnn`, `tcn`, `tabular`, `video_cnn`, plus a universal `hf_pipeline` wrapper that works with any HuggingFace model.
- **Pre-flight validation** — catches config mistakes (modality mismatches, AMP without CUDA, gated HF models, resource overruns, synthetic data into a tokenizer) before training spawns.
- **Resume / fine-tune from checkpoints** — `--resume` continues a run with optimizer + scheduler + epoch state intact; `--finetune` loads weights only with a fresh optimizer.
- **HuggingFace launcher** — paste a model id in the Predict tab, get a managed inference server backed by `--no-checkpoint` mode and per-server bearer-token auth.
- **Single source of truth for HF tasks** — every `pipeline_tag` maps to its `Auto*` class, processor type, input/output kind, generation policy, and coarse loss in `core/pipeline_specs.py`.
- **Live training** — Rich progress in the terminal, SSE event stream + interactive charts in the dashboard, and concurrent multi-run support.
- **Experiment tracking** — SQLite-backed history with status, best val loss, run durations, and a sortable Experiments tab.

---

## Install

```bash
git clone https://github.com/Derilius/neural-platform.git neural
cd neural
python3 -m venv .venv && source .venv/bin/activate
pip install -e .[dev]
```

The base install pulls PyTorch, FastAPI, Pydantic, Rich, scikit-learn — enough to train an MLP on synthetic data and run the dashboard. Modality-specific tooling lives in optional groups so you only install what you need:

| Group | Pulls in | Use when |
|-------|----------|----------|
| `[hf]` | `transformers`, `datasets`, `accelerate`, `tokenizers`, `huggingface-hub` | training or serving any HF model |
| `[vision]` | `torchvision` | image / video models, image inference preprocessing |
| `[audio]` | `torchaudio`, `soundfile`, `librosa`, `torchcodec` | audio models, ASR, HF audio datasets |
| `[video]` | `torchvision`, `av` | video models |
| `[peft]` | `peft` | LoRA adapter repos |
| `[tensorflow]`, `[jax]` | TF / JAX | alternative training adapters (stubbed today) |
| `[dev]` | `pytest`, `pytest-asyncio`, `black`, `ruff` | running the test suite |
| `[all]` | everything except alt frameworks | one-shot full install |

Quick combined install:

```bash
pip install -e .[hf,vision,audio,dev]
```

`pytest` should then collect all tests cleanly.

---

## CLI

```text
neural init       Scaffold a new experiment config
neural deps       Audit installed packages per modality
neural inspect    Inspect a HuggingFace dataset's columns
neural validate   Run the pre-flight validator on a config
neural train      Train a model (supports --resume / --finetune)
neural evaluate   Run val/test metrics on a checkpoint
neural serve      Launch the REST inference server
neural dashboard  Launch the web dashboard
neural list       List tracked experiments
neural status     Show one experiment's run history
neural export     Export a trained model to ONNX / TorchScript
```

### Train, then resume

```bash
# First run
neural train -c runs/imdb_lstm/config.yaml

# Continue from where it left off (full state)
neural train -c runs/imdb_lstm/config.yaml --resume runs/imdb_lstm/checkpoints/checkpoint_best.pt

# Adapt the same weights to a new task (weights only, fresh optimizer)
neural train -c runs/imdb_finetune/config.yaml --finetune runs/imdb_lstm/checkpoints/checkpoint_best.pt

# Shorthand: pick the best checkpoint by run name
neural train -c runs/imdb_lstm/config.yaml --resume-from-best imdb_lstm
```

### Serve an HF model with no local training

```bash
# Whisper-tiny ASR — weights pulled from HF, no checkpoint needed
neural serve -c runs/_hf_servers/whisper_tiny/config.yaml --no-checkpoint
```

(Or use the dashboard's Predict tab → **Launch from HuggingFace**, which synthesizes the config for you.)

### Validate before training

```bash
neural validate -c runs/cifar10_demo/config.yaml
# ✓ Config is valid: cifar10_demo (cnn)
```

The validator catches: missing HF dataset names, MLP/synthetic feature mismatches, regression with cross-entropy, gated HF models without a token, modality mismatches (Whisper paired with IMDB), AMP requested without CUDA, configs that won't fit in VRAM/RAM, and the `hf_pipeline + synthetic` mistake (synthesized server-only configs that aren't trainable).

---

## Web dashboard

```bash
neural dashboard
# open http://127.0.0.1:7860
```

Five tabs:

- **Builder** — visual config builder for all 9 model families with live YAML preview, dataset browser, and HF model browser with the resource-fit + compatibility inspector inline.
- **Train** — pick a config, kick off a run, get live SSE events, see all concurrent runs in a registry. Stop / forget per run.
- **Live** — single-run focus view: per-epoch chart, per-batch loss/lr, log tail.
- **Experiments** — sortable / filterable history with cascade delete.
- **Predict** — managed inference servers (start, stop, route through a secure proxy), launch from a saved config or directly from a HuggingFace model id, then send single-shot predictions with a latency chart.
- **Settings** — system telemetry, HF auth status, API docs link.

The Predict tab's **Launch from HuggingFace** option is the zero-shot path: paste a model id (e.g. `openai/whisper-tiny`), pick the pipeline task, and the manager spawns an inference subprocess with `--no-checkpoint`. The synthesized config sits under `runs/_hf_servers/` and is filtered out of the Train tab's listing.

---

## How configs work

```yaml
name: cifar10_demo
output_dir: runs

model:
  type: cnn
  framework: pytorch
  cnn:
    input_channels: 3
    input_height: 32
    input_width: 32
    output_size: 10
    conv_layers: [{out_channels: 32, kernel_size: 3}, {out_channels: 64, kernel_size: 3}]
    fc_layers: [{size: 128, dropout: 0.3}]

training:
  task: image_classification
  loss: cross_entropy
  num_epochs: 20
  batch_size: 64
  optimizer: { type: adamw, lr: 0.001 }
  scheduler: { type: cosine }
  device: auto

data:
  source: huggingface
  dataset_name: cifar10
```

The schema lives in `neural_platform/core/config.py` (Pydantic v2) and is the single source of truth for what the trainer, dashboard, and inference server expect. Every example under `configs/examples/` is a valid starting point.

---

## HuggingFace pipeline mapping

`core/pipeline_specs.py` is the single table that says, for every supported HF `pipeline_tag`:

- which `transformers.Auto*` class to instantiate
- which preprocessor to load (tokenizer / image processor / feature extractor / unified processor)
- what input shape the inference endpoint accepts
- whether to call `model.generate()` or just take logits
- which post-processing path renders the output (logits → top-k, generated tokens → decoded text, boxes, masks, depth, …)
- the coarse `core.config.Task` enum and default loss for the validator

So if you launch `openai/whisper-tiny` with `pipeline_task=automatic-speech-recognition`, the synthesizer + server agree: load `WhisperProcessor`, take audio waveform input, run `.generate()`, decode tokens, return text. No drift between the model wrapper, the synthesizer, and the input adapter.

---

## File layout

```
neural_platform/
├── core/
│   ├── config.py           # Pydantic schemas (single source of truth)
│   ├── trainer.py          # Unified training loop + resume/finetune
│   ├── validator.py        # Pre-flight checks
│   ├── pipeline_specs.py   # HF pipeline_tag → server requirements
│   ├── tasks.py            # Coarse Task taxonomy
│   ├── model_source.py     # HF Hub / local checkpoint discovery + inspector
│   ├── resource_fit.py     # RAM/VRAM/disk fit estimator
│   ├── hf_auth.py          # Token discovery + redaction (no token storage)
│   ├── evaluator.py        # Metric computation + Evaluator
│   ├── experiment.py       # SQLite experiment tracker
│   ├── event_bus.py        # JSONL event writer + async tail reader
│   ├── deps.py             # Dependency audit
│   ├── modality.py         # Modality detection helpers
│   └── tasks.py            # Task taxonomy
├── models/
│   ├── mlp.py, cnn.py, rnn.py, transformer.py
│   ├── audio.py, tcn.py, tabular.py, video.py
│   └── hf_pipeline.py      # Universal HF wrapper
├── frameworks/
│   ├── pytorch_adapter.py  # Primary backend
│   ├── tensorflow_adapter.py, jax_adapter.py   # Stubs
│   └── factory.py
├── data/
│   ├── loader.py           # Builds train/val/test loaders for every source
│   └── transforms.py
├── cli/
│   ├── commands.py         # Click CLI entry points
│   └── _templates.py       # `neural init` scaffolds
├── deploy/
│   └── server.py           # FastAPI inference server
└── web/
    ├── app.py              # FastAPI dashboard
    ├── inference_manager.py# Managed-server lifecycle (HF launcher lives here)
    ├── training_manager.py # Multi-run training subprocesses
    └── static/             # SPA: index.html + app.js + styles.css
```

---

## Tests

```bash
pip install -e .[dev]
pytest
```

The test suite is strictly offline (HF Hub calls and subprocess spawns are mocked). Coverage focuses on the failure modes that have actually broken users:

- Resume / fine-tune state restoration, scheduler/scaler round-trip, architecture-drift tolerance
- HF launcher: id validation, redirect handling (same-host follows, cross-origin rejected), `--no-checkpoint` subprocess wiring
- Validator rules: synthetic + hf_pipeline rejected, `_hf_servers/` configs filtered from the Train listing
- Pipeline specs: every UI-advertised task has a spec, ASR uses AutoProcessor + needs_generation, image classification uses AutoImageProcessor, /info returns 200 in no-checkpoint mode
- Inference manager lifecycle, bearer-token isolation, checkpoint resolution

---

## Roadmap notes

- The TensorFlow and JAX adapters are stubs today — the PyTorch path is the supported one.
- `video_cnn` is a basic 3D CNN baseline; production-quality video work would need purpose-built architectures (I3D, SlowFast).
- The HF pipeline table covers every standard `pipeline_tag`; tasks not in the table fall through to `AutoModel + AutoTokenizer` and the validator surfaces a warning.

---

## License

MIT. See `LICENSE` for details (or the license clause in `pyproject.toml`).
