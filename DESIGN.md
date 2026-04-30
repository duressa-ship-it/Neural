# NeuralForge — Design Document

> A self-hosted training platform that lets a user define, train, evaluate,
> and serve neural networks from a single config file — with a CLI for power
> users, a polished web dashboard for visibility, and a REST API for serving.

---

## 1. North Star

NeuralForge is a **playground-first** training environment. The product
question driving every architectural decision is:

> *Can a user go from "I have an idea for a model" to "it's trained, evaluated,
> and serving predictions" without ever leaving NeuralForge?*

Concretely:

- **Define** — a YAML config (or the visual Builder) is the only required
  input. Four model families (MLP, CNN, RNN, Transformer) cover most
  classification/regression tasks out of the box.
- **Train** — one command (`neural train`) or one button (the dashboard's
  Train page) launches a subprocess. Live metrics stream into the Live page
  via Server-Sent Events. SQLite persists every run for later comparison.
- **Evaluate** — `neural evaluate` runs metrics on val/test; the Curves page
  shows per-epoch loss/accuracy/MSE.
- **Serve** — `neural serve` starts a FastAPI inference server. The Predict
  page connects to it and runs live predictions, including class names and
  per-layer/weight introspection.

The platform is opinionated about a few things — config-driven everything,
PyTorch as the default framework, SQLite for tracking — and deliberately
*unopinionated* about others: any HuggingFace dataset, any torchvision
model, any compatible Python environment.

---

## 2. Repository Layout

```
neural/
├── neural_platform/
│   ├── core/
│   │   ├── config.py          ── Pydantic schemas: Experiment, Model, Training, Data, Deploy
│   │   ├── trainer.py         ── Unified training loop (early stopping, AMP, checkpoints, SSE events)
│   │   ├── evaluator.py       ── Metric accumulators + per-phase eval
│   │   ├── experiment.py      ── SQLite tracker (experiments, runs, metrics)
│   │   ├── event_bus.py       ── JSONL writer/async reader for live training events
│   │   ├── registry.py        ── Decorator-based plugin registry (models, optimizers…)
│   │   ├── validator.py       ── Pre-flight config validation
│   │   └── hf_introspect.py   ── HuggingFace feature/modality inspection (torch-free)
│   ├── data/
│   │   ├── loader.py          ── Unified DataLoader builder (CSV, image folder, HF, numpy, synthetic)
│   │   └── transforms.py      ── Config-driven preprocessing (image / tabular / text)
│   ├── models/
│   │   ├── base.py            ── BaseModel (count_parameters, save/load, summary)
│   │   ├── mlp.py             ── Feedforward network
│   │   ├── cnn.py             ── Custom CNN + torchvision backbones (resnet, vgg, efficientnet)
│   │   ├── rnn.py             ── LSTM / GRU / vanilla RNN
│   │   └── transformer.py     ── From-scratch encoder/encoder-decoder + HF wrapper
│   ├── frameworks/
│   │   ├── base.py            ── FrameworkAdapter ABC (train_step, eval_step, save/load_checkpoint)
│   │   ├── factory.py         ── get_adapter(config) → PyTorchAdapter | …
│   │   ├── pytorch_adapter.py ── PyTorch implementation (full)
│   │   ├── tensorflow_adapter.py ── TF stub (interface in place, ops TBD)
│   │   └── jax_adapter.py     ── JAX stub
│   ├── deploy/
│   │   └── server.py          ── FastAPI inference server (/predict, /info, /info/layers, /info/weights)
│   ├── web/
│   │   ├── app.py             ── Dashboard FastAPI app — 26 endpoints across 8 tags
│   │   └── static/
│   │       ├── index.html     ── SPA markup
│   │       ├── styles.css     ── Design tokens + components
│   │       └── app.js         ── Client logic (state, navigation, charts, SSE)
│   └── cli/
│       ├── commands.py        ── `neural {init,train,evaluate,serve,dashboard,list,status,
│       │                          export,validate,inspect}` Click group
│       └── _templates.py      ── YAML scaffolds per model family
├── configs/examples/          ── Annotated YAML configs for all 4 model families
├── runs/                      ── Per-experiment output dirs (config.yaml, checkpoints/, …)
│                                 + neuralforge.db, live_events.jsonl, train_subprocess.log
├── tests/
│   └── test_dataset_compat.py ── Validator + dataset-modality regression suite
└── pyproject.toml
```

The split into `core/`, `data/`, `models/`, `frameworks/`, `deploy/`, `web/`,
and `cli/` is deliberate: each module has at most one concern, and the
import graph fans **inward** from CLI → web/deploy → frameworks → models →
core. The validator and tests sit at the bottom of that graph, so they're
torch-free and run in milliseconds.

---

## 3. Core Concepts

### 3.1 The config object (`core/config.py`)

`ExperimentConfig` is the single object every other component takes as
input. It's a Pydantic v2 model, so:

- Loading is one call: `load_config(path)` → fully validated object.
- Schema validation happens before anything is built (`d_model %
  num_heads == 0`, `val_split + test_split < 1.0`, etc.).
- Round-tripping through `model_dump()` / `model_validate()` is lossless,
  so the dashboard's Builder UI can serialize → POST → save → re-load with
  no information loss.

Key sub-configs:

| Section          | Purpose                                                         |
| ---------------- | --------------------------------------------------------------- |
| `model.*`        | One of `mlp` / `cnn` / `rnn` / `transformer` blocks is populated |
| `training`       | Loss, optimizer, scheduler, batch size, AMP, early stopping      |
| `data`           | Source (CSV/image_folder/HF/numpy/synthetic) + per-source fields |
| `deploy`         | Inference server bind host/port, max batch size                  |

### 3.2 The framework adapter (`frameworks/base.py`)

`FrameworkAdapter` is the boundary between framework-agnostic logic
(trainer, evaluator) and the actual tensor library. It exposes:

```python
build_model() / build_optimizer() / build_scheduler() / build_loss()
get_device() / make_scaler()
train_step(model, batch, opt, loss_fn, scaler) -> (loss, metrics)
eval_step(model, batch, loss_fn)               -> (loss, metrics)
save_checkpoint(model, opt, path, extra)
load_checkpoint(path)                          -> (model, meta)
optimizer_step(model, opt, scaler)             -> ()  # for grad accumulation
```

`PyTorchAdapter` is the only fully implemented backend today.
`TensorFlowAdapter` and `JaxAdapter` are wired up to the registry but
delegate to stubs that raise `NotImplementedError` — the Builder UI still
lets a user select them, and the validator warns when chosen.

### 3.3 The training loop (`core/trainer.py`)

`Trainer.fit(train_loader, val_loader)` orchestrates:

1. **First-batch sanity check** — pulls one batch, validates input
   tensor shape vs. `model.{type}.input_size`. Catches modality
   mismatches (CNN → text dataset) in seconds, not minutes.
2. **Event stream open** — appends `training_start` to
   `<output_dir>/live_events.jsonl` (the SSE source).
3. **Tracker open** — `tracker.start_run()` returns a row id.
4. **Per-epoch loop**:
   - Training phase: `adapter.train_step` per batch, emit `batch` events
     every `log_every` batches.
   - Validation phase: `evaluator.evaluate`, emit `epoch` event.
   - Scheduler step (with `ReduceLROnPlateau` special-cased on val_loss).
   - Checkpoint save when `is_best` or `epoch % checkpoint_every == 0`,
     emit `checkpoint` event with class names embedded.
   - Early stopping on val loss with configurable patience.
5. **Finalize** — emit `training_end`, `tracker.finish_run()`,
   `tracker.update_experiment_status()`, print Rich summary table.

The `KeyboardInterrupt` path marks status as `interrupted` so partial runs
are still queryable via `neural list` and the Experiments page.

### 3.4 Live event bus (`core/event_bus.py`)

A JSONL file (`<output_dir>/live_events.jsonl`) is the simplest possible
inter-process queue we could justify:

- The trainer is a CLI subprocess (potentially launched by a user, by the
  dashboard, or by a future scheduler). It only knows how to write text.
- The dashboard tails the file from a different process. It only knows
  how to read text.
- No extra dependencies (Redis, RabbitMQ, Kafka), no port conflicts, and
  the file is itself an audit log.

Robustness measures:

- The reader detects file rotation (inode change) and truncation (size
  shrunk), so a fresh `neural train` truncating the file mid-stream
  doesn't break in-flight SSE clients.
- Partial line writes are held back until the next poll completes the line
  (no torn JSON parse).
- On dashboard startup we *only* mark stale `running` rows as
  `interrupted` if the events file has had no recent writes — otherwise a
  CLI training session running on the side gets clobbered.
- Subprocess crash detection: if `proc.poll()` returns non-None and there's
  no `training_end` line, the dashboard appends one synthetically with
  `status: 'failed'` and the exit code, so the Live UI can finalize.

Event schema:

```jsonc
{"type": "training_start", "ts": …, "experiment": "…", "model_type": "cnn", "framework": "pytorch", "total_epochs": 50, "total_batches": 313, "device": "mps"}
{"type": "batch", "epoch": 1, "batch": 10, "loss": 1.23, "metrics": {"accuracy": 0.5}, "lr": 1e-3, …}
{"type": "epoch", "epoch": 1, "train_metrics": {…}, "val_metrics": {…}, "lr": 1e-3, "elapsed": 12.4}
{"type": "checkpoint", "epoch": 5, "path": "…/checkpoint_best.pt", "is_best": true}
{"type": "early_stop", "epoch": 18, "best_epoch": 12, "best_val_loss": 0.45}
{"type": "training_end", "status": "completed|interrupted|failed", "best_epoch": 12, "best_val_loss": 0.45, "total_epochs": 18, "duration": 720.3, "exit_code": 0}
```

### 3.5 SQLite tracker (`core/experiment.py`)

Three tables: `experiments`, `runs`, `metrics`. WAL mode is on so the web
app can read while the trainer writes. Highlights:

- `interrupt_stale_runs()` — backfills `duration_secs` from `started_at`
  and pulls `best_val_loss` from logged epochs when crashes leave NULLs.
- `delete_experiment()` — cascade through runs and metrics so the
  Experiments drawer's Delete button is safe.
- `search_experiments(q, status)` — name/description LIKE + status filter,
  used by the search bar.

### 3.6 Pre-flight validator (`core/validator.py`)

Catches the kinds of errors that *should* fail at config time, not
30 seconds into a training run when a worker subprocess explodes. Used by:

- `neural train` (exit 2 with red errors before subprocess spawn)
- `neural validate` (standalone command, supports `--strict` and `--json`)
- `POST /api/train/start` (returns 422 with structured `{message,
  issues}` if it fails)
- `POST /api/configs/validate` (validate without saving)
- The dashboard's Train button — it parses 422 issues into a multi-line toast

Categories:

| Check                            | Where                  |
| -------------------------------- | ---------------------- |
| Identity / naming                 | `_validate_identity`   |
| Model arch invariants             | `_validate_model`      |
| Training params                   | `_validate_training`   |
| Per-source data presence + paths  | `_validate_data`       |
| Cross-cutting compat              | `_validate_data_model_compat` |
| HF dataset modality               | `_validate_hf_modality` |

For HF datasets we try `load_dataset_builder(name).info.features` — that's
a metadata-only fetch from the Hub, no data download — and inspect the
features to decide if the dataset is image / text / labeled. Falls back
to a name-based heuristic if `datasets` isn't installed or the dataset is
gated.

### 3.7 HuggingFace introspection (`core/hf_introspect.py`)

Pure-Python, torch-free, zero hard dependency on `datasets`. Duck-types
its way through a Features mapping using only `type(feat).__name__`,
`feat.names`, and `feat.dtype`. Output schema:

```python
{
    "columns":          [...],          # every column, in order
    "image_columns":    [...],          # HF Image features
    "text_columns":     [...],          # Value(string)
    "label_columns":    [...],          # ClassLabel + numeric named "label"/"target"/...
    "numeric_columns":  [...],          # everything else int/float
    "other_columns":    [...],
    "class_names":      [...] | None,   # ClassLabel.names from the first label
    "has_images":       bool,
    "has_text":         bool,
}
```

This single function powers: `neural inspect <name>`, the validator's
modality check, the loader's image-vs-text wrapper choice, and
`/api/hf/inspect` (which the dashboard Builder uses to auto-fill columns).

### 3.8 Data loading (`data/loader.py`)

The HF branch is now feature-aware. Decision tree:

```
data.source = huggingface, dataset_name = "X"
│
├── X is a torchvision builtin (mnist/cifar*/svhn/…)
│   └── _load_torchvision (cached, no Hub call)
│
└── else: load_dataset(X, split=…)
    │   inspect_features(...)
    │
    ├── model.type == "cnn", schema has images
    │   └── HuggingFaceImageDataset(image_col, label_col, transform)
    │
    ├── model.type == "transformer", schema has text
    │   └── HuggingFaceTextDataset(text_col, label_col, tokenizer)
    │
    └── modality mismatch → ValueError with hint
```

`_pick_column()` is the shared resolver: user pick wins (with
"is-this-actually-a-column" check), otherwise pick the first auto-detected
candidate, otherwise raise with the available column list in the error.

### 3.9 Inference server (`deploy/server.py`)

Built on FastAPI for free OpenAPI docs and request validation. Routes:

| Method | Route             | Purpose                                                   |
| ------ | ----------------- | --------------------------------------------------------- |
| GET    | `/health`         | Liveness + model-loaded flag + uptime                      |
| GET    | `/info`           | Model name, type, parameter counts, class names, epoch     |
| GET    | `/info/layers`    | Per-submodule type, parameters, shape, trainable flag      |
| GET    | `/info/weights`   | mean/std/min/max/sparsity per named parameter              |
| POST   | `/predict`        | Run inference; 422 on shape mismatch with hint              |
| POST   | `/predict/batch`  | Alias                                                      |
| GET    | `/docs`, `/redoc` | Auto-generated Swagger UI / ReDoc                          |

`PredictRequest` is flexible: the user supplies `inputs`, `tokens`, `text`,
or `image_b64` and the server picks the right tensor-builder for the
model type. Class names are returned alongside integer labels, so
predictions show "tulip [id 3]" instead of just "3".

### 3.10 Dashboard server (`web/app.py`)

26 endpoints organized into 8 tags:

| Tag           | Endpoints                                                                      |
| ------------- | ------------------------------------------------------------------------------ |
| System        | `/api/health`, `/api/system`, `/api/stats`                                      |
| Experiments   | `/api/experiments[/{id}/{search/interrupt/DELETE}]` (5 routes)                  |
| Metrics       | `/api/experiments/{id}/metrics`, `/api/runs/{id}/metrics`                       |
| Configs       | `/api/configs`, `/api/configs/{load,save,validate}`, `/api/hf/inspect`          |
| Training      | `/api/train/{status,start,stop,cleanup,logs}`                                   |
| Checkpoints   | `/api/checkpoints`, `/api/checkpoints/recent`                                   |
| Inference     | `/api/proxy/{health,info,predict}` — CORS-safe forwarding to a serve server     |
| Live          | `/api/training/live`, `/api/events/stream` (SSE)                                |

Subprocess management is the trickiest piece: see §3.4 above for crash
detection. The `_kill_process_group` helper sends SIGTERM to the entire
process group spawned with `start_new_session=True`, then SIGKILL after a
6-second grace period — this catches PyTorch DataLoader worker children
that would otherwise survive the parent.

### 3.11 SPA frontend (`web/static/`)

Three files now (was one monolith):

- `index.html` (~42 KB) — pure markup; defines the eight pages, the side
  drawer, the command palette, and the toast container.
- `styles.css` (~24 KB) — design tokens (`--bg`, `--primary`, `--font-mono`,
  …) + every component class.
- `app.js` (~70 KB) — `State` object, `navigate()`, page initializers
  (`initOverview`, `initTrain`, `initLive`, `initBuilder`, `initPredict`,
  …), Chart.js wrappers, SSE client, Cmd-K palette.

There's no build step — vanilla ES, Chart.js from a CDN, fonts from
Google Fonts. The `/static` mount serves them straight from disk so a
hot reload during development is just F5.

Pages (eight, sidebar-navigated):

1. **Overview** — KPI cards, system telemetry strip, recent experiments,
   best-val-loss bar chart, recent checkpoints, quick-action grid.
2. **Train** — config picker, override editor, controls, log tail.
3. **Live** — real-time KPIs, batch + epoch + accuracy charts, event log,
   ETA. Subscribes to `/api/events/stream`.
4. **Builder** — visual config builder for all four model families, with
   live YAML preview that posts to `/api/configs/save`.
5. **Experiments** — sortable / filterable / searchable table; clicking a
   row opens the side drawer with runs, mini-chart, and Delete.
6. **Curves** — full-size training curves with linear/log Y-axis toggle.
7. **Checkpoints** — list with serve / evaluate / export commands.
8. **Predict** — connect to a server, run inference; shows top-K bars
   with class names + raw JSON drawer + rolling latency chart.

### 3.12 CLI (`cli/commands.py`)

```
neural init       --model {mlp|cnn|rnn|transformer} --name <slug>
neural validate   -c <path>  [--strict] [--json]
neural inspect    <hf_dataset_name>
neural train      -c <path>  [-O key=value ...]
neural evaluate   -c <path>  [--checkpoint <path>] [--split val|test|train]
neural serve      -c <path>  [--checkpoint <path>] [--host …] [--port …]
neural dashboard  [--output-dir runs] [--port 7860]
neural list       [--output-dir runs]
neural status     <experiment_id>
neural export     -c <path>  [--format onnx|torchscript]
```

Click is the framework. `--override` (or `-O`) on `train` deserves special
mention: it accepts dotted-path key=value mutations of the loaded config
(e.g. `-O training.optimizer.lr=5e-4`) and re-validates the resulting
config through Pydantic before training, so users can sweep
hyperparameters without ever touching the YAML.

---

## 4. Data Flow Diagrams

### 4.1 A typical training session

```
┌────────────┐   POST /api/train/start    ┌────────────────┐
│ Dashboard  │──────────────────────────▶ │  Dashboard API │
│   (Train)  │                            │  (web/app.py)  │
└────────────┘                            └─────┬──────────┘
                                                │
                                  spawn (start_new_session=True)
                                                │
                                                ▼
              ┌─────────────────────────────────────────────────┐
              │  neural train --config … --override …           │
              │   → Trainer.fit(train_loader, val_loader)       │
              │      ├── adapter.build_model / .build_optimizer │
              │      ├── for epoch in range(num_epochs):        │
              │      │    ├── train_step → events.batch         │
              │      │    ├── eval_step  → events.epoch         │
              │      │    ├── checkpoint → events.checkpoint    │
              │      │    └── tracker.log_metrics (SQLite)      │
              │      └── events.training_end                    │
              └────────────┬────────────────────────────────────┘
                           │
            writes JSONL ▼ │ ▼ writes SQLite
                ┌────────────┐ ┌─────────────────┐
                │ live_events│ │ neuralforge.db  │
                │   .jsonl   │ │                 │
                └─────┬──────┘ └─────┬───────────┘
                      │              │
        tail (async)  │              │  SELECT
                      ▼              ▼
              ┌─────────────────────────────┐
              │  GET /api/events/stream     │  ── SSE ──▶  Live page
              │  GET /api/experiments       │  ── JSON ──▶ Experiments
              └─────────────────────────────┘
```

### 4.2 An inference request

```
Predict page  ── POST /api/proxy/predict ──▶  Dashboard API
                                              │
                                              │ httpx
                                              ▼
                                   ┌──────────────────────┐
                                   │  Inference server    │
                                   │  (deploy/server.py)  │
                                   │                      │
                                   │  _build_input(req)   │
                                   │       ↓              │
                                   │  model(inputs)       │
                                   │       ↓              │
                                   │  _build_predictions  │
                                   │  (with class_names)  │
                                   └──────────┬───────────┘
                                              │
                                              ▼
              { predictions: [[{label, class_name, probability, score}, …]],
                model_type: …, latency_ms: … }
                                              │
        Dashboard normalizes (canonical shape)│
                                              ▼
         ┌─────────────────────────────────────────┐
         │  Top-K bars + class names               │
         │  Rolling latency chart                  │
         │  Raw JSON drawer                        │
         └─────────────────────────────────────────┘
```

---

## 5. Conventions

- **Docstrings** are user-facing. Every public function explains *what to
  pass in*, *what comes back*, *what errors to expect*. Internal helpers
  get a one-liner.
- **Errors** carry hints. A `ValueError` from the data loader doesn't just
  say "missing column" — it lists the available columns and the likely
  candidates.
- **No magic strings inside hot paths**. Enums (`ModelType`, `Optimizer`,
  `LossFunction`, `DataSource`, `Task`) are everywhere.
- **Pydantic models are the contract**. If a value crosses a process
  boundary, it's serialized through Pydantic.
- **No global state** except the dashboard's process-management dict, which
  is keyed by `state[…]` so it's easy to mock in tests.
- **The trainer is framework-agnostic**. All PyTorch-specific logic lives
  in `frameworks/pytorch_adapter.py`. To add JAX support, fill in
  `frameworks/jax_adapter.py` — no other module changes.

---

## 6. Extension Points

| Goal                          | Where to plug in                                              |
| ----------------------------- | ------------------------------------------------------------- |
| New model family              | Subclass `models.base.BaseModel`, decorate with `@registry.register(MODEL, "name")`, add a `<name>: <Type>Config` block to `ModelConfig`. |
| New optimizer / scheduler     | Add an enum value, extend `_build_optimizer` / `_build_scheduler` in `pytorch_adapter.py`. |
| New data source               | Add a `DataSource` enum value, add a branch in `build_dataloaders`, add a Dataset subclass. |
| New metric                    | Edit `_compute_metrics` in the framework adapter.             |
| New framework                 | Implement `FrameworkAdapter`, register in `frameworks/factory.py`. |
| New dashboard page            | Add a `<section class="page" id="page-X">` in `index.html`, an `initX()` in `app.js`, a `data-page="X"` nav-item. |
| New API endpoint              | Add a route in `web/app.py` with a tag, a summary, and proper `responses=…`. |

---

## 7. Test Strategy

- `tests/test_dataset_compat.py` — validator + dataset-modality regression
  suite. Runs in <100 ms with no network. Includes:
  - `TestFeatureInspection` — duck-typed fixture datasets to verify
    `inspect_features` categorizes columns correctly across image / text /
    labeled / unlabeled / multi-column shapes.
  - `TestValidatorCoreCases` — every error the validator catches (missing
    paths, missing target columns, regression+cross_entropy, MLP feature
    mismatch, from-scratch transformer with synthetic data).
  - `TestKnownDatasetHeuristics` — name-based fallback when `datasets`
    isn't installed (cnn+imdb=error, transformer+cifar10=error,
    cnn+cifar10=clean).
  - `TestOnlineHF` — gated behind `--online`; hits the real Hub.
- Future additions: `tests/test_inference_shapes.py` for `/predict` shape
  validation, `tests/test_event_bus.py` for the truncation/inode-rotation
  guarantees of the SSE reader.

---

## 8. Roadmap

### Near-term (next round)

- **HuggingFace dataset browser plugin** — `/api/hf/search?q=` endpoint
  proxying the Hub search API; Builder UI with thumbnail / row-count /
  config previews; one-click "use this dataset" pinning.
- **Hyperparameter sweeps** — accept a sweep-spec YAML (grid or random),
  spawn N trainers, surface a sweep view in the dashboard.
- **Confusion matrix + per-class metrics** in the Curves page (when class
  names are available).
- **Real-time gradient-norm chart** in Live (cheap; emit per-batch).

### Medium-term

- **TensorFlow + JAX adapters fully implemented** — currently stubs.
- **Distributed training** — at least DDP on a single node behind a
  `training.distributed` config block.
- **Model registry** — push trained checkpoints to a HuggingFace Hub
  repository or a local model-card directory.
- **Plugin architecture for datasets** — make `data.source` extensible
  via entry points, so a third party can ship a `neural-plugin-kaggle`
  package and have the Builder pick it up automatically.

### Long-term

- **Multi-tenant dashboard** — auth, per-user workspaces, shared models.
- **Notebook-friendly imports** — `from neural_platform import quick_train`
  for one-line fits in a Jupyter cell, no YAML required.

---

## 9. Operational Notes

### Files written during a run

```
runs/<experiment>/
├── config.yaml              ── frozen at training time
├── checkpoints/
│   ├── checkpoint_best.pt   ── lowest val loss
│   └── checkpoint_epoch_NNNN.pt
├── training.log             ── (CLI redirect)
└── ../
    ├── neuralforge.db       ── shared across all experiments in this output_dir
    ├── live_events.jsonl    ── overwritten on every new run
    └── train_subprocess.log ── overwritten on every dashboard-launched run
```

### The two ports

- **7860** — dashboard (`neural dashboard`).
- **8080** — inference server (`neural serve`). The dashboard's Predict
  page connects to this URL by default.

These are independent processes; the dashboard uses an httpx proxy to
talk to the inference server (avoids browser CORS issues even though the
inference server has CORS open — convenient for end users).

### Known limitations

- `pin_memory=True` in DataLoader only when CUDA is available. MPS users
  get pageable memory transfers (which is correct).
- Mixed precision (`training.mixed_precision`) is CUDA-only. The validator
  warns when it's set on a non-CUDA device.
- The transformer's from-scratch path requires tokenized inputs. The HF
  loader auto-resolves a tokenizer (`bert-base-uncased` by default), but
  pairing the from-scratch transformer with a non-text dataset is flagged
  by the validator.
- `predict` for regression returns a single Prediction with the sigmoid
  probability — for true regression you want the raw `score` field.
