"""
NeuralForge Inference Server
FastAPI REST API for serving trained neural network models.

Endpoints
─────────
GET  /health              Liveness + model load status
GET  /info                Model metadata, parameter counts, class names
GET  /info/layers         Per-layer parameter & shape breakdown
GET  /info/weights        Weight statistics (mean/std/min/max per parameter)
POST /predict             Run inference on a single sample (returns top-K)
POST /predict/batch       Same, but for a list of samples
GET  /docs                Swagger UI (FastAPI built-in)
GET  /redoc               ReDoc (FastAPI built-in)

Input shape contract
────────────────────
MLP / RNN     {"inputs":[float,...]}            or batched {"inputs":[[...],[...]]}
CNN           {"image_b64":"<base64>"}           or {"inputs":[[[...]]]}
Transformer   {"tokens":[int,...]}               or {"text":"..."} (server tokenizes)

Response shape
──────────────
{
  "predictions": [[{label, probability, class_name?}, ...]],
  "model_type":  "mlp",
  "latency_ms":  1.2
}
The outer list is per-sample; the inner list is top-K within that sample.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel as PydanticModel, Field

from neural_platform.core.config import ExperimentConfig


# ---------------------------------------------------------------------------
# Request / Response schemas (with rich examples for the /docs page)
# ---------------------------------------------------------------------------

class PredictRequest(PydanticModel):
    """
    Flexible prediction request — supply *one* of these fields based on
    your model type:

    * **inputs**:    flat list of floats (MLP, RNN), or a list of timesteps
    * **tokens**:    list of token IDs (transformer, when bypassing the tokenizer)
    * **image_b64**: base64-encoded image bytes (CNN)
    * **text**:      raw string (transformer with HuggingFace tokenizer)

    Add `top_k` to control how many predictions you get back.
    """
    inputs: Optional[Union[List[float], List[List[float]], List[List[List[float]]]]] = Field(
        None, description="Flat float vector, batch of vectors, or 3D tensor (CNN)"
    )
    tokens: Optional[Union[List[int], List[List[int]]]] = Field(
        None, description="Token IDs for transformer models"
    )
    attention_mask: Optional[Union[List[int], List[List[int]]]] = Field(
        None, description="Attention mask for transformer models (1=token, 0=pad)"
    )
    image_b64: Optional[str] = Field(
        None, description="Base64-encoded image bytes (PNG/JPG/GIF) for CNN models"
    )
    text: Optional[str] = Field(
        None, description="Raw text — server tokenizes via HuggingFace AutoTokenizer"
    )
    context: Optional[str] = Field(
        None,
        description=(
            "Passage / context paragraph for question-answering and similar "
            "text-pair tasks. The server pairs `text` (the question) with "
            "`context` (the passage) when the loaded model's pipeline_task "
            "is question-answering or text-pair-shaped."
        ),
    )
    candidate_labels: Optional[List[str]] = Field(
        None,
        description=(
            "Candidate labels for zero-shot classification. Pass alongside "
            "`text` for text zero-shot or `image_b64` for CLIP-style image "
            "zero-shot. The server runs NLI / contrastive matching across "
            "the labels and returns scores per label."
        ),
    )
    top_k: int = Field(5, ge=1, le=100, description="How many top predictions to return")
    return_probabilities: bool = Field(True, description="Include softmax probabilities")
    # Generation knobs (used for ASR / summarization / translation / image-to-text /
    # text-generation / image-text-to-text / any-to-any). All optional —
    # the server falls back to the model's HF generation config defaults.
    max_new_tokens: Optional[int] = Field(
        None, ge=1, le=4096,
        description="Cap the number of generated tokens. Defaults to the model's HF generation config.",
    )
    temperature: Optional[float] = Field(
        None, ge=0.0, le=2.0, description="Sampling temperature for generative tasks."
    )
    do_sample: Optional[bool] = Field(
        None, description="Enable sampling for generative tasks (default: greedy)."
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "summary": "MLP / tabular",
                    "value": {"inputs": [0.1, 0.5, 0.9, 0.2, 0.7], "top_k": 3},
                },
                {
                    "summary": "Transformer text",
                    "value": {"text": "the movie was great", "top_k": 5},
                },
                {
                    "summary": "Transformer pre-tokenized",
                    "value": {"tokens": [101, 2023, 2003, 1037, 102], "top_k": 5},
                },
            ],
        }
    }


class Prediction(PydanticModel):
    """One prediction. `class_name` is populated when the checkpoint includes class labels."""
    label: int = Field(..., description="Class index (0-based)")
    class_name: Optional[str] = Field(None, description="Human-readable class name, if known")
    probability: Optional[float] = Field(None, description="Softmax probability (0–1)")
    score: Optional[float] = Field(None, description="Raw logit value")


class PredictResponse(PydanticModel):
    """Top-K predictions for each input sample."""
    predictions: List[List[Prediction]] = Field(..., description="One inner list per sample")
    model_type: str
    latency_ms: float = Field(..., description="Server-side inference latency in ms")


class HealthResponse(PydanticModel):
    status: str = Field(..., description="'ok' when ready to serve")
    model_loaded: bool
    device: str
    uptime_secs: float


class LayerInfo(PydanticModel):
    name: str
    type: str
    parameters: int
    trainable: bool
    shape: Optional[List[int]] = None


class InfoResponse(PydanticModel):
    model_name: str
    model_type: str
    framework: str
    parameter_count: Optional[int] = None
    trainable_parameters: Optional[int] = None
    checkpoint_path: Optional[str] = Field(
        None,
        description=(
            "Absolute path to the .pt checkpoint backing this server. "
            "None when the server was launched in `--no-checkpoint` mode "
            "(hf_pipeline serving HF weights directly)."
        ),
    )
    device: str
    class_names: Optional[List[str]] = Field(
        None, description="Human-readable class labels indexed by class id"
    )
    output_size: Optional[int] = None
    epoch: Optional[int] = Field(None, description="Epoch the loaded checkpoint was saved at")
    val_loss: Optional[float] = Field(None, description="Best validation loss at checkpoint time")
    # Populated when the server is hosting an HF-pipeline model: the HF id,
    # pipeline_task, and the spec-derived auto/processor classes the server
    # actually loaded. Lets the UI distinguish "trained run" vs "zero-shot HF".
    pipeline_task:    Optional[str] = None
    hf_model_id:      Optional[str] = None
    auto_class:       Optional[str] = None
    processor_class:  Optional[str] = None


class LayersResponse(PydanticModel):
    total_parameters: int
    trainable_parameters: int
    layers: List[LayerInfo]


class WeightStat(PydanticModel):
    name: str
    shape: List[int]
    numel: int
    mean: float
    std: float
    min: float
    max: float
    sparsity: float = Field(..., description="Fraction of values close to zero (|w| < 1e-6)")


class WeightsResponse(PydanticModel):
    summary: List[WeightStat]


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

def create_inference_app(config: ExperimentConfig,
                         checkpoint_path: Optional[str] = None) -> FastAPI:
    """
    Build a FastAPI inference server for a trained NeuralForge model.

    The server loads the model on startup, exposes /predict, and surfaces
    rich metadata via /info, /info/layers, and /info/weights.

    ``checkpoint_path`` may be ``None`` for **HuggingFace pipeline** servers —
    the model wrapper calls ``from_pretrained()`` at build time, so the
    weights come from the Hub (or HF cache) rather than from a NeuralForge
    checkpoint. This is the path the dashboard's "Launch from HF" button
    uses for zero-shot inference. Any other model type with no checkpoint
    fails the startup hook with a clear error.

    **Authentication.** When the `NEURAL_INFERENCE_TOKEN` env var is set,
    every endpoint *except* `/health` and `/docs` requires
    `Authorization: Bearer <token>`. The dashboard manager generates a
    random token per launched server and holds it server-side; the token
    never appears in API responses or logs. When the env var is unset
    (e.g. a manually started `neural serve`) no auth is enforced — same
    behaviour as before.

    **CORS.** Restricted to localhost origins by default (override via
    `NEURAL_INFERENCE_CORS_ORIGINS`).
    """
    import os as _os
    import secrets as _secrets
    expected_token = (_os.environ.get("NEURAL_INFERENCE_TOKEN") or "").strip() or None
    # Default-to-secure: when no token was provided (e.g. `neural serve` was
    # run manually without piping one in), generate a one-shot token,
    # log it once, and require it. The opt-out is `NEURAL_INFERENCE_AUTH=off`
    # — set it explicitly when you really mean "I trust everyone on this
    # network", e.g. behind a reverse proxy that handles auth.
    auth_mode = (_os.environ.get("NEURAL_INFERENCE_AUTH") or "on").strip().lower()
    if not expected_token and auth_mode != "off":
        expected_token = _secrets.token_urlsafe(32)
        # Print to stderr ONCE so the operator can grab the token but the
        # value never goes through the normal logger / structured logs.
        import sys as _sys
        print(
            f"[NeuralForge] No NEURAL_INFERENCE_TOKEN was set — generated a "
            f"one-shot token for this server.\n"
            f"  Use:  Authorization: Bearer {expected_token}\n"
            f"  Or set NEURAL_INFERENCE_AUTH=off to disable auth (NOT "
            f"recommended on shared networks).",
            file=_sys.stderr, flush=True,
        )

    app = FastAPI(
        title=f"NeuralForge — {config.model.name}",
        description=(
            f"Inference server for **{config.model.type.value.upper()}** model "
            f"`{config.model.name}` ({config.model.framework.value}).\n\n"
            "See `/info` for model metadata, `/info/layers` for the architecture "
            "breakdown, and `/predict` to run inference. "
            "Send `Accept: application/json` and a body matching `PredictRequest`."
            + ("\n\n**Auth required:** include `Authorization: Bearer <token>` "
               "on every request except `/health`. Set `NEURAL_INFERENCE_AUTH=off` "
               "or pass `--no-auth` to `neural serve` to disable this." if expected_token else "")
        ),
        version="0.3.1",
        contact={"name": "NeuralForge"},
        # Themed swagger UI mounted explicitly below.
        docs_url=None,
        redoc_url=None,
        openapi_tags=[
            {"name": "System",     "description": "Liveness, model metadata, weight introspection."},
            {"name": "Inference",  "description": "Run predictions on one or more samples."},
        ],
    )

    # ---- Themed Swagger / ReDoc (matches the dashboard's dark theme) ----
    from fastapi.responses import HTMLResponse as _HTMLResp
    _DOCS_CSS = """
      <style>
        :root { color-scheme: dark; }
        html, body { background: #0d0e10; color: #e6e8ea; }
        .swagger-ui, .swagger-ui .topbar { background: #0d0e10 !important; }
        .swagger-ui .info { background: #131418 !important; }
        .swagger-ui h1, .swagger-ui h2, .swagger-ui h3, .swagger-ui h4,
        .swagger-ui .info p, .swagger-ui .info .title { color: #e6e8ea !important; }
        .swagger-ui .opblock { background: #131418 !important; border-color: #23262d !important; }
        .swagger-ui .opblock .opblock-summary { background: #1a1d22 !important; }
        .swagger-ui input, .swagger-ui textarea, .swagger-ui select {
          background: #0d0e10 !important; color: #e6e8ea !important; border-color: #2c3140 !important; }
        .swagger-ui .topbar-wrapper img { display: none; }
        .swagger-ui .topbar-wrapper::before {
          content: "NeuralForge Inference Server";
          color: #e6e8ea; font-weight: 600; font-size: 14px; padding: 10px 14px;
        }
      </style>
    """
    @app.get("/docs", include_in_schema=False)
    async def _themed_docs() -> _HTMLResp:
        return _HTMLResp(f"""<!doctype html><html><head>
<meta charset="utf-8"/><title>NeuralForge Inference</title>
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/swagger-ui.css"/>
{_DOCS_CSS}</head><body>
<div id="swagger-ui"></div>
<script src="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/swagger-ui-bundle.js"></script>
<script>window.ui = SwaggerUIBundle({{url:'/openapi.json', dom_id:'#swagger-ui', deepLinking:true}});</script>
</body></html>""")
    cors_env = (_os.environ.get("NEURAL_INFERENCE_CORS_ORIGINS") or "").strip()
    cors_origins = [o.strip() for o in cors_env.split(",") if o.strip()] or [
        "http://localhost", "http://127.0.0.1", "http://[::1]",
        "http://localhost:8000", "http://127.0.0.1:8000",
        "http://localhost:8765", "http://127.0.0.1:8765",
    ]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_origin_regex=r"^http://(localhost|127\.0\.0\.1|\[::1\])(:\d+)?$",
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
    )

    if expected_token:
        # Bearer-token middleware. Constant-time comparison to avoid
        # leaking via timing differences. The token is read from env and
        # compared in-process — never logged, never returned.
        import hmac as _hmac
        from fastapi import Request as _Request
        from fastapi.responses import JSONResponse as _JSONResponse

        @app.middleware("http")
        async def _require_bearer(request: _Request, call_next):
            path = request.url.path or ""
            # Health + docs exempted so manager polling and Swagger work.
            if path in ("/health", "/docs", "/redoc", "/openapi.json"):
                return await call_next(request)
            auth = request.headers.get("authorization") or ""
            if not auth.lower().startswith("bearer "):
                return _JSONResponse(status_code=401,
                                     content={"detail": "Bearer token required."})
            sent = auth.split(None, 1)[1].strip()
            if not _hmac.compare_digest(sent, expected_token):
                return _JSONResponse(status_code=401,
                                     content={"detail": "Bearer token rejected."})
            return await call_next(request)

    from neural_platform.frameworks.factory import get_adapter
    adapter = get_adapter(config)
    device = adapter.get_device()

    container: Dict[str, Any] = {"started_at": time.time()}

    @app.on_event("startup")
    async def load_model():
        if checkpoint_path:
            model, meta = adapter.load_checkpoint(checkpoint_path)
        else:
            # No-checkpoint mode: the only legitimate use is hf_pipeline,
            # whose model wrapper resolves weights via `from_pretrained` at
            # __init__ time. For every other model type this means "untrained
            # random weights" — useless for inference, so we refuse it.
            mtype = config.model.type.value
            if mtype != "hf_pipeline":
                raise RuntimeError(
                    f"Inference server started with no checkpoint, but model "
                    f"type is '{mtype}' — only 'hf_pipeline' models can be "
                    "served without a NeuralForge checkpoint (their weights "
                    "load from HuggingFace at startup)."
                )
            model = adapter.build_model()
            meta = {}
        model.eval()
        container["model"] = model
        container["device"] = device
        container["meta"] = meta
        container["config"] = config
        container["checkpoint"] = checkpoint_path
        # Spec-aware preprocessor: tokenizer / feature extractor / image
        # processor / unified processor depending on the configured pipeline_task.
        # Returned object has the same .from_pretrained-style interface
        # consumers care about; the input adapter handles the type dispatch.
        processor = _try_load_processor(config)
        container["processor"] = processor
        # Keep the legacy "tokenizer" key around so any third-party code
        # poking at the container still finds it.
        container["tokenizer"] = processor

    # ------------------------------------------------------------------
    # System
    # ------------------------------------------------------------------

    @app.get("/health", response_model=HealthResponse, tags=["System"], summary="Liveness check")
    async def health():
        """Returns server health and whether the model has finished loading."""
        return HealthResponse(
            status="ok",
            model_loaded="model" in container,
            device=str(device),
            uptime_secs=round(time.time() - container["started_at"], 2),
        )

    @app.get("/info", response_model=InfoResponse, tags=["System"], summary="Model metadata")
    async def info():
        """Model name, type, parameter counts, checkpoint epoch, class labels."""
        model = container.get("model")
        meta = container.get("meta", {}) or {}
        n_params = (model.count_parameters(trainable_only=False)
                     if hasattr(model, "count_parameters") else None)
        n_trainable = (model.count_parameters(trainable_only=True)
                        if hasattr(model, "count_parameters") else None)

        # Output size depends on the model family. For HF pipelines we also
        # try to read num_labels off the loaded model's config — the user
        # didn't have to set it in the synthesized config when launching,
        # but classifiers do expose it on the HF side.
        output_size = None
        try:
            arch = config.model.get_arch_config()
            output_size = getattr(arch, "output_size", None)
        except Exception:
            pass
        if output_size is None and config.model.type.value == "hf_pipeline":
            try:
                hf_cfg = getattr(getattr(model, "encoder", None), "config", None)
                if hf_cfg is not None:
                    output_size = getattr(hf_cfg, "num_labels", None)
            except Exception:
                pass

        # Spec-derived metadata for HF pipelines so the UI can show what
        # auto-class / processor the server actually loaded.
        pipeline_task = config.training.pipeline_task
        hf_id = (config.model.hf_pipeline.pretrained
                  if config.model.hf_pipeline else None)
        auto_class = None
        processor_class = None
        if config.model.type.value == "hf_pipeline":
            try:
                from neural_platform.core.pipeline_specs import (
                    resolve as _resolve_spec, PROCESSOR_AUTO_CLASS,
                )
                spec = _resolve_spec(pipeline_task)
                auto_class = spec.auto_class
                processor_class = PROCESSOR_AUTO_CLASS.get(spec.processor_kind)
            except Exception:
                pass

        # If the HF model carries id2label, surface it as class_names so the
        # Predict UI shows real labels instead of bare ids. Honor any class
        # names that were stashed on the checkpoint at training time first
        # (those override — they were chosen by the user).
        class_names = meta.get("class_names")
        if class_names is None and config.model.type.value == "hf_pipeline":
            try:
                hf_cfg = getattr(getattr(model, "encoder", None), "config", None)
                id2label = getattr(hf_cfg, "id2label", None) if hf_cfg else None
                if id2label and len(id2label):
                    class_names = [id2label[i] for i in sorted(id2label.keys())]
            except Exception:
                pass

        return InfoResponse(
            model_name=config.model.name,
            model_type=config.model.type.value,
            framework=config.model.framework.value,
            parameter_count=n_params,
            trainable_parameters=n_trainable,
            checkpoint_path=checkpoint_path,
            device=str(device),
            class_names=class_names,
            output_size=output_size,
            epoch=meta.get("epoch"),
            val_loss=meta.get("val_loss"),
            pipeline_task=pipeline_task,
            hf_model_id=hf_id,
            auto_class=auto_class,
            processor_class=processor_class,
        )

    @app.get("/info/layers", response_model=LayersResponse, tags=["System"], summary="Per-layer breakdown")
    async def info_layers():
        """Walk the model and return parameter counts and shapes per submodule."""
        model = container.get("model")
        if model is None:
            raise HTTPException(503, "Model not loaded")
        layers: List[LayerInfo] = []
        for name, module in model.named_modules():
            if name == "" or list(module.children()):
                continue  # skip the root and intermediate containers
            params = list(module.parameters(recurse=False))
            n_params = sum(p.numel() for p in params)
            if n_params == 0:
                continue
            shape = None
            for p in params:
                shape = list(p.shape)
                break
            trainable = any(p.requires_grad for p in params)
            layers.append(LayerInfo(
                name=name,
                type=type(module).__name__,
                parameters=n_params,
                trainable=trainable,
                shape=shape,
            ))
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        return LayersResponse(total_parameters=total, trainable_parameters=trainable, layers=layers)

    @app.get("/info/weights", response_model=WeightsResponse, tags=["System"], summary="Weight statistics")
    async def info_weights():
        """
        Mean / std / min / max / sparsity per named parameter. Useful for
        spotting dead layers (zero std), saturated tanh blocks, or aggressive
        weight decay.
        """
        model = container.get("model")
        if model is None:
            raise HTTPException(503, "Model not loaded")
        out: List[WeightStat] = []
        with torch.no_grad():
            for name, p in model.named_parameters():
                t = p.detach().float().cpu()
                out.append(WeightStat(
                    name=name,
                    shape=list(p.shape),
                    numel=p.numel(),
                    mean=float(t.mean().item()),
                    std=float(t.std().item() if t.numel() > 1 else 0.0),
                    min=float(t.min().item()),
                    max=float(t.max().item()),
                    sparsity=float((t.abs() < 1e-6).float().mean().item()),
                ))
        return WeightsResponse(summary=out)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    @app.post("/predict", response_model=PredictResponse, tags=["Inference"],
              summary="Run inference on a single sample",
              responses={
                  422: {"description": "Input validation failed — check shape & required fields."},
                  503: {"description": "Model not loaded yet."},
              })
    async def predict(request: PredictRequest):
        return await _do_predict(request)

    @app.post("/predict/batch", response_model=PredictResponse, tags=["Inference"],
              summary="Same as /predict — accepts batched inputs")
    async def predict_batch(request: PredictRequest):
        return await _do_predict(request)

    # ------------------------------------------------------------------
    # Internal predict runner
    # ------------------------------------------------------------------

    async def _do_predict(request: PredictRequest) -> PredictResponse:
        model = container.get("model")
        if model is None:
            raise HTTPException(503, "Model not loaded")
        meta = container.get("meta", {}) or {}
        class_names = meta.get("class_names")
        # Allow id2label from the HF model config to act as class_names if
        # the user didn't pin any at training time. The Predict UI uses
        # this to render real labels instead of bare integers.
        if class_names is None and config.model.type.value == "hf_pipeline":
            try:
                hf_cfg = getattr(getattr(model, "encoder", None), "config", None)
                id2label = getattr(hf_cfg, "id2label", None) if hf_cfg else None
                if id2label and len(id2label):
                    class_names = [id2label[i] for i in sorted(id2label.keys())]
            except Exception:
                pass

        t0 = time.time()
        model_type = config.model.type.value

        try:
            tensor_input = _build_input(
                request, model_type, device, config, container.get("processor"),
            )
        except _InputError as e:
            raise HTTPException(status_code=422, detail=str(e))
        except Exception as e:
            raise HTTPException(status_code=422, detail=f"Input error: {e}")

        # Resolve the spec ONCE — drives both the generation branch and the
        # predictions render below. For non-hf_pipeline models, spec is
        # the default (POSTPROC_LOGITS, no generation).
        from neural_platform.core.pipeline_specs import (
            resolve as _resolve_spec, POSTPROC_GENERATED_TEXT,
            POSTPROC_QA_SPANS, POSTPROC_BOXES, POSTPROC_MASKS, POSTPROC_DEPTH,
            POSTPROC_TOKEN_LOGITS,
        )
        spec = (_resolve_spec(config.training.pipeline_task)
                if model_type == "hf_pipeline"
                else _resolve_spec(None))

        with torch.no_grad():
            try:
                # ---- generative path (.generate + decode) -----------------
                # Whisper / BART / T5 / GPT / image-to-text / image-text-to-text
                # / any-to-any. Calling the forward pass and softmaxing the
                # logits gives token-step probabilities — useless for the
                # user. Run .generate() on the underlying HF encoder and
                # decode via the loaded processor.
                if spec.needs_generation:
                    encoder = getattr(model, "encoder", model)
                    proc = container.get("processor")
                    if not hasattr(encoder, "generate"):
                        raise HTTPException(
                            status_code=500,
                            detail=f"Task '{spec.task}' needs .generate() but the "
                                   "loaded model doesn't expose it. Pick a different "
                                   "auto-class or task.",
                        )
                    gen_inputs = (tensor_input if isinstance(tensor_input, dict)
                                  else {"inputs": tensor_input})
                    # Forward generation kwargs from the request.
                    gen_kwargs: Dict[str, Any] = {}
                    if request.max_new_tokens is not None:
                        gen_kwargs["max_new_tokens"] = int(request.max_new_tokens)
                    if request.temperature is not None:
                        gen_kwargs["temperature"] = float(request.temperature)
                    if request.do_sample is not None:
                        gen_kwargs["do_sample"] = bool(request.do_sample)
                    generated = encoder.generate(**gen_inputs, **gen_kwargs)
                    text = None
                    if proc is not None:
                        try:
                            decoded = proc.batch_decode(
                                generated, skip_special_tokens=True,
                            )
                            text = decoded[0] if decoded else None
                        except Exception:
                            # Fall back to .tokenizer.decode for processors that
                            # don't expose batch_decode (rare, older HF).
                            try:
                                tok = getattr(proc, "tokenizer", proc)
                                text = tok.decode(generated[0], skip_special_tokens=True)
                            except Exception:
                                text = None
                    latency_ms = (time.time() - t0) * 1000
                    # Surface the decoded text as a single Prediction with
                    # `class_name` carrying the string. The Predict UI
                    # already knows how to render that field.
                    label_str = text if text is not None else "(decoder failed)"
                    pred = Prediction(
                        label=0,
                        class_name=label_str,
                        probability=None,
                        score=None,
                    )
                    return PredictResponse(
                        predictions=[[pred]],
                        model_type=model_type,
                        latency_ms=round(latency_ms, 2),
                    )

                # ---- non-generative path (existing behavior) -------------
                if isinstance(tensor_input, dict):
                    raw_output = model(**tensor_input)
                elif isinstance(tensor_input, (list, tuple)):
                    raw_output = model(*tensor_input)
                else:
                    raw_output = model(tensor_input)
            except HTTPException:
                raise
            except RuntimeError as e:
                # Most common: shape mismatch. Report as 422 (caller's fault) with
                # the model's expected dim baked in if we can derive it.
                hint = _shape_hint(config)
                raise HTTPException(
                    status_code=422,
                    detail=f"Shape mismatch in inference: {e}{hint}",
                )
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Inference error: {e}")

        # ---- structured-output post-processing ---------------------------
        # The HF wrapper returns either a tensor of logits (the common
        # case) or the full HF output object for tasks that need multiple
        # tensors (QA = start_logits + end_logits; object detection = boxes
        # + scores; depth = predicted_depth; segmentation = logits + masks).
        # Branch on spec.output_kind so each task gets a coherent response
        # shape rather than 500ing on a missing .dim() attribute.
        if spec.output_kind == POSTPROC_QA_SPANS and not isinstance(raw_output, torch.Tensor):
            return _qa_response(raw_output, tensor_input, container.get("processor"),
                                 model_type, t0)

        if spec.output_kind == POSTPROC_BOXES and not isinstance(raw_output, torch.Tensor):
            return _boxes_response(raw_output, model_type, t0,
                                    class_names_fn=lambda i: _lookup_class(class_names, i))

        if spec.output_kind == POSTPROC_DEPTH and not isinstance(raw_output, torch.Tensor):
            return _depth_response(raw_output, model_type, t0)

        if spec.output_kind == POSTPROC_MASKS and not isinstance(raw_output, torch.Tensor):
            return _masks_response(raw_output, model_type, t0)

        # Anything else: treat the output as logits. If we still got a
        # structured output (e.g. an unknown HF output shape), pull
        # `.logits` off it — the wrapper would normally do that, but a
        # custom auto class might bypass the wrapper.
        if isinstance(raw_output, torch.Tensor):
            logits = raw_output
        elif hasattr(raw_output, "logits"):
            logits = raw_output.logits
        elif hasattr(raw_output, "last_hidden_state"):
            logits = raw_output.last_hidden_state
        else:
            raise HTTPException(
                status_code=500,
                detail=f"Model returned a {type(raw_output).__name__} the server "
                       f"doesn't know how to render. Task '{spec.task}' may need a "
                       "purpose-built postprocessor — file an issue with the model id.",
            )

        latency_ms = (time.time() - t0) * 1000

        if logits.dim() == 1:
            logits = logits.unsqueeze(0)

        predictions: List[List[Prediction]] = []
        for sample_logits in logits:
            sample_preds = _build_predictions(
                sample_logits, request.top_k, request.return_probabilities, class_names,
            )
            predictions.append(sample_preds)

        return PredictResponse(
            predictions=predictions,
            model_type=model_type,
            latency_ms=round(latency_ms, 2),
        )

    return app


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _InputError(Exception):
    pass


def _try_load_processor(config: ExperimentConfig):
    """Best-effort preprocessor load.

    Returns the right transformers preprocessor for the configured task —
    a tokenizer for text models, a feature extractor for raw audio, an
    image processor for vision models, or a unified processor for
    multimodal models (Whisper, BLIP, …). Falls back to a tokenizer for
    backward compatibility with transformer model configs.

    Driven by :mod:`core.pipeline_specs` so the choice stays consistent
    with the auto-class the model wrapper will load.
    """
    mtype = config.model.type.value
    if mtype not in ("transformer", "hf_pipeline"):
        return None
    try:
        import transformers
    except ImportError:
        return None

    # Choose the model id to point the processor at. Mirrors the same
    # discovery logic the old tokenizer loader used.
    name = None
    if mtype == "transformer" and config.model.transformer and config.model.transformer.use_pretrained:
        name = config.model.transformer.use_pretrained
    if mtype == "hf_pipeline" and config.model.hf_pipeline and config.model.hf_pipeline.pretrained:
        name = config.model.hf_pipeline.pretrained
    if not name and config.data.transforms and isinstance(config.data.transforms, dict):
        name = config.data.transforms.get("text", {}).get("tokenizer")
    name = name or "bert-base-uncased"

    kwargs: Dict[str, Any] = {}
    # Some processors require trust_remote_code (e.g. custom HF repos).
    if (mtype == "hf_pipeline" and config.model.hf_pipeline
            and config.model.hf_pipeline.trust_remote_code):
        kwargs["trust_remote_code"] = True
    if (mtype == "hf_pipeline" and config.model.hf_pipeline
            and config.model.hf_pipeline.revision):
        kwargs["revision"] = config.model.hf_pipeline.revision
    # Pass HF token through if we have one — gated processors (e.g. Llama)
    # need it. Token never leaves this process; we read via hf_auth.
    try:
        from neural_platform.core.hf_auth import get_token as _hf_token
        t = _hf_token()
        if t:
            kwargs["token"] = t
    except Exception:
        pass

    # Resolve the processor class via the spec table.
    from neural_platform.core.pipeline_specs import (
        resolve as _resolve_spec, PROCESSOR_AUTO_CLASS,
        PROCESSOR_TOKENIZER, PROCESSOR_NONE,
    )
    if mtype == "hf_pipeline":
        spec = _resolve_spec(config.training.pipeline_task)
        processor_kind = spec.processor_kind
    else:
        # Plain `transformer` configs are always tokenizer-shaped.
        processor_kind = PROCESSOR_TOKENIZER

    if processor_kind == PROCESSOR_NONE:
        return None

    auto_name = PROCESSOR_AUTO_CLASS.get(processor_kind, "AutoTokenizer")
    auto_cls = getattr(transformers, auto_name, None)

    # If the chosen class doesn't exist (transformers is too old), or the
    # load fails (model only ships some processor types), fall back through
    # the chain: AutoProcessor → AutoFeatureExtractor → AutoImageProcessor →
    # AutoTokenizer. This keeps something useful loaded even when the spec
    # picked the "ideal" class but the model exposes a narrower one.
    fallbacks = [auto_name, "AutoProcessor", "AutoFeatureExtractor",
                 "AutoImageProcessor", "AutoTokenizer"]
    seen = set()
    for fb_name in fallbacks:
        if fb_name in seen:
            continue
        seen.add(fb_name)
        cls = getattr(transformers, fb_name, None)
        if cls is None:
            continue
        try:
            return cls.from_pretrained(name, **kwargs)
        except Exception:
            continue
    return None


# Backward-compat alias — older callers / tests still import _try_load_tokenizer.
_try_load_tokenizer = _try_load_processor


def _shape_hint(config: ExperimentConfig) -> str:
    """Append a "model expects shape X" hint to error messages."""
    try:
        mt = config.model.type.value
        arch = config.model.get_arch_config()
        if mt == "mlp":
            return f"\nHint: this MLP expects exactly {arch.input_size} features per sample."
        if mt == "cnn":
            return (f"\nHint: this CNN expects images of shape "
                    f"({arch.input_channels}, {arch.input_height}, {arch.input_width}).")
        if mt == "rnn":
            return f"\nHint: this RNN expects {arch.input_size} features per timestep."
        if mt == "transformer":
            return f"\nHint: max_seq_len={arch.max_seq_len}, vocab_size={arch.vocab_size}."
        if mt == "audio_cnn":
            samples = int(arch.sample_rate * arch.duration_secs)
            return (f"\nHint: this audio model expects {samples} waveform samples "
                    f"({arch.sample_rate}Hz × {arch.duration_secs}s) per clip.")
        if mt == "tcn":
            return f"\nHint: this TCN expects {arch.input_size} channels per timestep."
        if mt == "tabular":
            return (f"\nHint: this tabular model expects {len(arch.numeric_features)} numeric "
                    f"+ {len(arch.categorical_features)} categorical features.")
        if mt == "video_cnn":
            return (f"\nHint: this video model expects {arch.num_frames} frames at "
                    f"({arch.input_channels}, {arch.input_height}, {arch.input_width}).")
    except Exception:
        pass
    return ""


def _build_input(request: PredictRequest, model_type: str, device, config: ExperimentConfig, tokenizer):
    """Convert a PredictRequest into the correct tensor format."""

    if model_type == "mlp":
        data = request.inputs
        if data is None:
            raise _InputError("'inputs' field is required for MLP models")
        if not isinstance(data, list) or not data:
            raise _InputError("'inputs' must be a non-empty list of floats")
        if not isinstance(data[0], (list, tuple)):
            data = [data]
        # Strict shape check vs config
        expected = config.model.mlp.input_size
        actual = len(data[0])
        if actual != expected:
            raise _InputError(
                f"MLP input has {actual} features, but model expects {expected}. "
                f"Pad or truncate to {expected}, or retrain with input_size={actual}."
            )
        return torch.tensor(data, dtype=torch.float32).to(device)

    if model_type == "rnn":
        data = request.inputs
        if data is None:
            raise _InputError("'inputs' field is required for RNN models — pass a list of timesteps")
        # Accept (timesteps, features), (features,) for single timestep, or batched
        if not data:
            raise _InputError("'inputs' is empty")
        # Normalize to (batch, timesteps, features)
        if isinstance(data[0], (int, float)):
            data = [[[float(v)] for v in data]]                  # scalar sequence
        elif isinstance(data[0], list) and (not data[0] or isinstance(data[0][0], (int, float))):
            data = [data]                                         # (T, F) → batch of 1
        # else assume already (batch, T, F)
        return torch.tensor(data, dtype=torch.float32).to(device)

    if model_type == "cnn":
        if request.image_b64 is not None:
            import base64
            import io
            try:
                from PIL import Image
                import torchvision.transforms as T
            except ImportError as exc:
                raise _InputError(f"CNN inference needs Pillow + torchvision: {exc}")
            img_bytes = base64.b64decode(request.image_b64)
            img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            arch = config.model.cnn
            transform = T.Compose([
                T.Resize((arch.input_height, arch.input_width)),
                T.ToTensor(),
            ])
            return transform(img).unsqueeze(0).to(device)
        if request.inputs is not None:
            data = request.inputs
            if not isinstance(data[0], (list, tuple)):
                data = [data]
            return torch.tensor(data, dtype=torch.float32).to(device)
        raise _InputError(
            "CNN requires 'image_b64' (base64-encoded PNG/JPG) or 'inputs' (raw tensor)."
        )

    if model_type == "transformer":
        if request.tokens is None and request.text is None:
            raise _InputError(
                "Transformer needs 'tokens' (list of int IDs) or 'text' (string for the server "
                "to tokenize). 'text' requires the server to have a tokenizer loaded."
            )
        if request.tokens is None and request.text is not None:
            if tokenizer is None:
                raise _InputError(
                    "No tokenizer is loaded on this server, so 'text' input can't be processed. "
                    "Pre-tokenize on the client and send 'tokens' instead."
                )
            enc = tokenizer(
                request.text,
                max_length=config.model.transformer.max_seq_len,
                padding="max_length", truncation=True, return_tensors="pt",
            )
            return {
                "input_ids": enc["input_ids"].to(device),
                "attention_mask": enc["attention_mask"].to(device),
            }
        tokens = request.tokens
        if not isinstance(tokens[0], (list, tuple)):
            tokens = [tokens]
        # Validate IDs vs vocab_size
        vocab = config.model.transformer.vocab_size
        flat = [t for row in tokens for t in row]
        if flat and (max(flat) >= vocab or min(flat) < 0):
            raise _InputError(
                f"Token IDs out of range — vocab_size={vocab}, but request includes "
                f"min={min(flat)}, max={max(flat)}."
            )
        input_ids = torch.tensor(tokens, dtype=torch.long).to(device)
        if request.attention_mask is not None:
            mask_data = request.attention_mask
            if not isinstance(mask_data[0], (list, tuple)):
                mask_data = [mask_data]
            mask = torch.tensor(mask_data, dtype=torch.long).to(device)
            return {"input_ids": input_ids, "attention_mask": mask}
        return {"input_ids": input_ids}

    if model_type == "audio_cnn":
        # Accepts a flat list of waveform samples in `inputs`.
        # Pads/truncates to (sample_rate × duration_secs).
        data = request.inputs
        if data is None:
            raise _InputError(
                "'audio_cnn' models need 'inputs' as a list of waveform samples. "
                "Sample rate must match config (default 16000)."
            )
        if not isinstance(data, list) or not data:
            raise _InputError("'inputs' must be a non-empty list of floats (waveform samples)")
        if isinstance(data[0], (list, tuple)):
            wav = torch.tensor(data, dtype=torch.float32)            # batched (B, samples)
        else:
            wav = torch.tensor([data], dtype=torch.float32)          # single (1, samples)
        arch = config.model.audio_cnn
        target = int(arch.sample_rate * arch.duration_secs)
        if wav.size(1) < target:
            wav = torch.nn.functional.pad(wav, (0, target - wav.size(1)))
        elif wav.size(1) > target:
            wav = wav[:, :target]
        return wav.to(device)

    if model_type == "tcn":
        data = request.inputs
        if data is None or not data:
            raise _InputError("'tcn' models need 'inputs' as a list of timesteps")
        # Accept: flat scalar sequence, (T, F) for single sample, or (B, T, F).
        if isinstance(data[0], (int, float)):
            data = [[[float(v)] for v in data]]
        elif isinstance(data[0], list) and (not data[0] or isinstance(data[0][0], (int, float))):
            data = [data]
        return torch.tensor(data, dtype=torch.float32).to(device)

    if model_type == "tabular":
        # Allow either flat numeric vector via `inputs` or richer structured
        # input via `inputs={"numeric":[...], "categorical":[...]}`.
        data = request.inputs
        if data is None:
            raise _InputError(
                "'tabular' models need 'inputs' — either a flat list of numeric "
                "features or {'numeric': [...], 'categorical': [...]}."
            )
        if isinstance(data, dict):
            num = data.get("numeric") or []
            cat = data.get("categorical") or []
            num_t = torch.tensor([num], dtype=torch.float32).to(device) if num else torch.zeros(1, 0).to(device)
            cat_t = torch.tensor([cat], dtype=torch.long).to(device) if cat else torch.zeros(1, 0, dtype=torch.long).to(device)
            return {"numeric": num_t, "categorical": cat_t}
        if isinstance(data, list) and data and isinstance(data[0], (int, float)):
            return torch.tensor([data], dtype=torch.float32).to(device)
        raise _InputError("'tabular' inputs must be a list of numbers or a dict with numeric/categorical keys")

    if model_type == "video_cnn":
        # Accepts a 4D nested list: (T, C, H, W) for single video, or 5D for batch.
        data = request.inputs
        if data is None:
            raise _InputError(
                "'video_cnn' needs 'inputs' as a 4D nested list (T, C, H, W) or 5D for batch. "
                "Most users will preprocess clips client-side."
            )
        t = torch.tensor(data, dtype=torch.float32)
        if t.dim() == 4:               # (T, C, H, W) → (1, C, T, H, W)
            t = t.permute(1, 0, 2, 3).unsqueeze(0)
        elif t.dim() == 5 and t.shape[1] != config.model.video_cnn.input_channels:
            # (B, T, C, H, W) → (B, C, T, H, W)
            t = t.permute(0, 2, 1, 3, 4)
        return t.to(device)

    if model_type == "hf_pipeline":
        # The universal HF wrapper handles every modality; route the request
        # to whichever input shape matches the resolved pipeline_task.
        return _build_hf_pipeline_input(request, config, device, tokenizer)

    raise _InputError(f"Unknown model type: {model_type}")


def _build_hf_pipeline_input(request: PredictRequest, config: ExperimentConfig,
                              device, processor):
    """Dispatch on the configured `pipeline_task`'s spec.

    Reads the spec's ``input_kind`` from :mod:`core.pipeline_specs`. The
    same table the synthesizer + auto-class lookup use, so an HF launch
    that sets ``pipeline_task=automatic-speech-recognition`` always lands
    on the audio path, regardless of how the user typed it.

    Returns whatever shape the underlying HF model's forward / generate
    expects — usually a dict from a ``processor(...)`` call (covers both
    tokenizer-style and feature-extractor-style preprocessors).
    """
    from neural_platform.core.pipeline_specs import (
        resolve as _resolve_spec,
        INPUT_TEXT, INPUT_TEXT_PAIR, INPUT_TEXT_LABELS,
        INPUT_IMAGE, INPUT_IMAGE_LABELS, INPUT_IMAGE_TEXT,
        INPUT_AUDIO, INPUT_AUDIO_TEXT, INPUT_VIDEO, INPUT_TENSOR,
        INPUT_ANY,
    )
    spec = _resolve_spec(config.training.pipeline_task)
    kind = spec.input_kind

    # ----- text family ---------------------------------------------------
    if kind in (INPUT_TEXT, INPUT_TEXT_PAIR, INPUT_TEXT_LABELS):
        if request.tokens is not None:
            tokens = request.tokens
            if not isinstance(tokens[0], (list, tuple)):
                tokens = [tokens]
            return {"input_ids": torch.tensor(tokens, dtype=torch.long).to(device)}
        if request.text is None:
            raise _InputError(
                f"Task '{spec.task}' needs 'text' (server-tokenized) or 'tokens' (pre-tokenized)."
            )
        if processor is None:
            raise _InputError(
                "No tokenizer is loaded on this server. Pre-tokenize on the "
                "client and send 'tokens' instead, or restart with a model "
                "whose tokenizer is on the Hub."
            )
        # For QA / text-pair tasks the user sends `text` (question) +
        # `context` (passage). HF tokenizers accept these as a positional
        # pair, which is what the model's forward expects.
        try:
            if kind == INPUT_TEXT_PAIR and request.context:
                enc = processor(
                    request.text, request.context,
                    max_length=384,
                    padding="max_length", truncation=True, return_tensors="pt",
                )
            else:
                enc = processor(
                    request.text,
                    max_length=128,
                    padding="max_length", truncation=True, return_tensors="pt",
                )
        except Exception as exc:
            raise _InputError(
                f"The processor for {spec.task} couldn't tokenize the input: {exc}. "
                "Try sending pre-tokenized `tokens` instead."
            )
        return {k: v.to(device) for k, v in enc.items()}

    # ----- any-to-any: dispatch on whichever fields the request carries ---
    if kind == INPUT_ANY:
        # Unified multimodal LMs (Gemma-3, Qwen2-VL, Phi-3.5-vision) accept
        # whatever combination the user supplies — text alone, image alone,
        # image+text, audio+text, etc. We hand the bundle to the processor
        # and let it route by what's present.
        if processor is None:
            raise _InputError(
                "Any-to-any models need a loaded HF processor (AutoProcessor). "
                "Restart the server with a model whose processor is on the Hub."
            )
        proc_kwargs: Dict[str, Any] = {}
        if request.text is not None:
            proc_kwargs["text"] = request.text
        if request.image_b64 is not None:
            import base64, io
            from PIL import Image
            proc_kwargs["images"] = Image.open(
                io.BytesIO(base64.b64decode(request.image_b64))
            ).convert("RGB")
        if request.inputs is not None:
            # Treat numeric `inputs` as raw audio waveform — most common
            # any-to-any audio path.
            wav = request.inputs
            if isinstance(wav, list) and wav and isinstance(wav[0], list):
                wav = wav[0]
            proc_kwargs["audio"] = [float(x) for x in wav]
            proc_kwargs.setdefault("sampling_rate", 16000)
        if not proc_kwargs:
            raise _InputError(
                f"Task '{spec.task}' needs at least one of `text`, `image_b64`, "
                "or `inputs` (audio waveform)."
            )
        proc_kwargs["return_tensors"] = "pt"
        try:
            enc = processor(**proc_kwargs)
        except Exception as exc:
            raise _InputError(
                f"Any-to-any processor rejected the input bundle: {exc}. "
                "Verify the model's expected modalities and try again."
            )
        # Some processors return a dict-like BatchEncoding; some return
        # plain dicts. Both are iterable as dict so we treat them uniformly.
        return {k: (v.to(device) if hasattr(v, "to") else v) for k, v in dict(enc).items()}

    # ----- image family --------------------------------------------------
    if kind == INPUT_IMAGE:
        if request.image_b64 is None:
            raise _InputError(
                f"Task '{spec.task}' needs 'image_b64' — base64-encoded PNG/JPG bytes."
            )
        import base64, io
        try:
            from PIL import Image
        except ImportError as exc:
            raise _InputError(f"Image inference needs Pillow: {exc}")
        img = Image.open(io.BytesIO(base64.b64decode(request.image_b64))).convert("RGB")
        # Prefer the model's own image processor (handles correct mean/std,
        # resize, normalization). Fall back to a generic 224×224 only when
        # no processor loaded — most HF vision models accept that shape.
        if processor is not None and hasattr(processor, "__call__"):
            try:
                enc = processor(images=img, return_tensors="pt")
                return {k: v.to(device) for k, v in enc.items()}
            except Exception:
                pass   # fall through to generic transform
        try:
            import torchvision.transforms as T
        except ImportError as exc:
            raise _InputError(f"Image inference fallback needs torchvision: {exc}")
        transform = T.Compose([T.Resize((224, 224)), T.ToTensor()])
        return transform(img).unsqueeze(0).to(device)

    if kind == INPUT_IMAGE_TEXT:
        if request.image_b64 is None or request.text is None:
            raise _InputError(
                f"Task '{spec.task}' needs both 'image_b64' and 'text' "
                "(the question / instruction)."
            )
        if processor is None:
            raise _InputError(
                "Multimodal task needs a loaded HF processor (tokenizer + "
                "image processor combined). Restart with the model on the Hub."
            )
        import base64, io
        from PIL import Image
        img = Image.open(io.BytesIO(base64.b64decode(request.image_b64))).convert("RGB")
        try:
            enc = processor(images=img, text=request.text, return_tensors="pt")
        except Exception as exc:
            raise _InputError(f"Multimodal preprocessor rejected the input: {exc}")
        return {k: v.to(device) for k, v in enc.items()}

    if kind == INPUT_IMAGE_LABELS:
        # CLIP-style zero-shot — image + candidate text labels.
        if request.image_b64 is None:
            raise _InputError("Zero-shot image classification needs 'image_b64'.")
        if processor is None:
            raise _InputError("CLIP-style models need a loaded HF processor.")
        labels = request.text   # accept comma-separated string or list of strings
        if not labels:
            raise _InputError(
                "Zero-shot image classification needs candidate labels in the "
                "'text' field, comma-separated (e.g. 'a cat, a dog, a banana')."
            )
        if isinstance(labels, str):
            labels = [s.strip() for s in labels.split(",") if s.strip()]
        import base64, io
        from PIL import Image
        img = Image.open(io.BytesIO(base64.b64decode(request.image_b64))).convert("RGB")
        try:
            enc = processor(images=img, text=labels, return_tensors="pt", padding=True)
        except Exception as exc:
            raise _InputError(f"CLIP processor rejected the input: {exc}")
        return {k: v.to(device) for k, v in enc.items()}

    # ----- audio family --------------------------------------------------
    if kind in (INPUT_AUDIO, INPUT_AUDIO_TEXT):
        if request.inputs is None:
            raise _InputError(
                f"Task '{spec.task}' needs 'inputs' as a list of waveform samples "
                "(default sample rate 16000)."
            )
        data = request.inputs
        # Flatten to a 1D float waveform — feature extractors expect that.
        if isinstance(data, list) and data and isinstance(data[0], (int, float)):
            wav = [float(x) for x in data]
        elif isinstance(data, list) and data and isinstance(data[0], (list, tuple)):
            wav = [float(x) for x in data[0]]   # take first sample if 2D was sent
        else:
            wav = list(data)

        # If we have a feature extractor / processor, run it. ASR + audio
        # classification with HF processors expect mel-spectrograms, not
        # raw waveform — skipping this is what causes "got MPSFloatType
        # instead of Long" for Whisper (forward gets a raw waveform tensor
        # but expects pre-extracted features).
        if processor is not None and hasattr(processor, "__call__"):
            try:
                # Most feature extractors expect kwarg `sampling_rate`.
                enc = processor(
                    wav, sampling_rate=16000, return_tensors="pt",
                )
                return {k: v.to(device) for k, v in enc.items()}
            except Exception as exc:
                # Some processors don't accept `sampling_rate` (older
                # versions); fall back to the no-kwarg call.
                try:
                    enc = processor(wav, return_tensors="pt")
                    return {k: v.to(device) for k, v in enc.items()}
                except Exception:
                    raise _InputError(
                        f"The audio processor rejected the waveform: {exc}. "
                        "Verify the model's expected sample rate (most HF "
                        "audio models want 16 kHz)."
                    )
        # No feature extractor available — fall back to raw waveform tensor
        # (works for older audio_cnn-style models that take raw inputs).
        return torch.tensor([wav], dtype=torch.float32).to(device)

    # ----- video, fallback ----------------------------------------------
    if kind == INPUT_VIDEO:
        if request.inputs is None:
            raise _InputError(
                f"Task '{spec.task}' needs 'inputs' as a 4D nested list (T, C, H, W)."
            )
        t = torch.tensor(request.inputs, dtype=torch.float32)
        if t.dim() == 4:
            t = t.unsqueeze(0)
        return t.to(device)

    # Generic tensor fallback — last resort.
    if request.inputs is None:
        raise _InputError(
            f"Task '{spec.task}' needs 'inputs' (list of floats) or one of "
            "'text' / 'tokens' / 'image_b64' depending on the model's modality."
        )
    data = request.inputs
    if isinstance(data, list) and data and isinstance(data[0], (int, float)):
        return torch.tensor([data], dtype=torch.float32).to(device)
    return torch.tensor(data, dtype=torch.float32).to(device)


def _build_predictions(
    logits: torch.Tensor,
    top_k: int,
    return_probs: bool,
    class_names: Optional[List[str]],
) -> List[Prediction]:
    """Convert raw logits into a list of Prediction objects.

    Robust to whatever shape the upstream model returns — token-classification
    models can emit ``(seq_len, num_classes)``, single-class HF heads emit
    ``(1, num_classes)`` even after the batch dim is iterated over, and
    legacy classifiers emit a 1-D ``(num_classes,)``. The previous
    implementation dispatched on ``size(0) == 1`` (the leading dim) which
    spuriously matched single-token / single-batch multi-class outputs and
    crashed inside ``torch.sigmoid(logits).item()`` with "a Tensor with N
    elements cannot be converted to Scalar".

    The fix:
      * Squeeze leading singleton dims so a ``(1, 1, num_classes)`` or
        ``(1, num_classes)`` collapses to ``(num_classes,)`` before we
        decide.
      * For token-classification (``(seq_len, num_classes)`` after squeeze),
        average over the sequence axis to produce a single distribution —
        not perfect but actionable, and the top-k is meaningful.
      * Dispatch on ``numel() == 1`` to detect the *true* scalar branch
        (binary / regression with a single-output head). Multi-class always
        runs softmax + topk regardless of leading singletons.
    """
    # Drop any leading singleton dims so we work with a flat class vector.
    while logits.dim() > 1 and logits.size(0) == 1:
        logits = logits.squeeze(0)

    # Token-level classification: (seq_len, num_classes). Average across
    # the sequence axis to surface document-level top-k. The dedicated
    # POSTPROC_TOKEN_LOGITS path in the server runs *before* this for
    # callers that want per-token output.
    if logits.dim() >= 2:
        logits = logits.mean(dim=tuple(range(logits.dim() - 1)))

    if logits.numel() == 1:
        # True scalar — single-output regression head or binary
        # classification with a single logit. Sigmoid → probability.
        score = float(logits.flatten()[0].item())
        prob = float(torch.sigmoid(logits.flatten()).item())
        label = int(prob > 0.5)
        return [Prediction(
            label=label,
            class_name=_lookup_class(class_names, label),
            probability=round(prob, 6) if return_probs else None,
            score=round(score, 6),
        )]

    # Multi-class: softmax + topk over the class axis.
    probs = torch.softmax(logits, dim=-1)
    k = min(top_k, logits.size(-1))
    top_probs, top_labels = probs.topk(k)
    return [
        Prediction(
            label=int(top_labels[i].item()),
            class_name=_lookup_class(class_names, int(top_labels[i].item())),
            probability=round(float(top_probs[i].item()), 6) if return_probs else None,
            score=round(float(logits[int(top_labels[i].item())].item()), 6),
        )
        for i in range(k)
    ]


def _lookup_class(class_names: Optional[List[str]], idx: int) -> Optional[str]:
    if not class_names:
        return None
    if 0 <= idx < len(class_names):
        return class_names[idx]
    return None


# ---------------------------------------------------------------------------
# Structured-output postprocessors
# ---------------------------------------------------------------------------
# Each of these takes the bare HF output object that the wrapper passed
# through (start_logits + end_logits for QA, pred_boxes + scores for
# object detection, etc.) and turns it into a PredictResponse the Predict
# UI can render. Surface area is deliberately the same as the
# logits-based path: a single-sample top-K list with `class_name`
# carrying the human-readable answer.

def _qa_response(outputs, tensor_input, processor, model_type: str,
                 t0: float) -> "PredictResponse":
    """Decode a question-answering structured output to a span string.

    HF QA models return ``QuestionAnsweringModelOutput(start_logits,
    end_logits, …)``. The "answer" is the contiguous token span between
    argmax(start) and argmax(end) — we slice that out of the original
    ``input_ids`` and decode it back to text via the tokenizer half of the
    processor (or the processor itself when it's a plain tokenizer).
    """
    start_logits = outputs.start_logits
    end_logits = outputs.end_logits
    # Take the first (and only — we batch=1) sample.
    s_logits = start_logits[0] if start_logits.dim() > 1 else start_logits
    e_logits = end_logits[0] if end_logits.dim() > 1 else end_logits
    start = int(torch.argmax(s_logits).item())
    end = int(torch.argmax(e_logits).item())
    if end < start:
        # Models occasionally pick spans in reverse; clamp gracefully.
        start, end = end, start
    # Confidence: softmax(start) * softmax(end). Useful for ranking, even
    # if not strictly calibrated.
    s_prob = float(torch.softmax(s_logits, dim=-1)[start].item())
    e_prob = float(torch.softmax(e_logits, dim=-1)[end].item())
    confidence = s_prob * e_prob

    # Decode the span — input_ids live in the tensor_input dict.
    answer = "(no input_ids in request)"
    input_ids = tensor_input.get("input_ids") if isinstance(tensor_input, dict) else None
    if input_ids is not None and processor is not None:
        try:
            tokenizer = getattr(processor, "tokenizer", processor)
            ids = input_ids[0] if input_ids.dim() > 1 else input_ids
            span = ids[start:end + 1]
            answer = tokenizer.decode(span, skip_special_tokens=True)
        except Exception as exc:
            answer = f"(decode failed: {exc})"

    latency_ms = (time.time() - t0) * 1000
    return PredictResponse(
        predictions=[[Prediction(
            label=0,
            class_name=answer.strip() or "(empty span)",
            probability=round(confidence, 6),
            score=None,
        )]],
        model_type=model_type,
        latency_ms=round(latency_ms, 2),
    )


def _boxes_response(outputs, model_type: str, t0: float, class_names_fn
                     ) -> "PredictResponse":
    """Render object-detection output as one Prediction per detected box.

    Shape: each Prediction's ``class_name`` carries
    ``"<label> @ x1,y1,x2,y2"`` so the existing Predict UI can render
    something useful without a custom widget. ``probability`` carries
    the detection score.
    """
    pred_boxes = getattr(outputs, "pred_boxes", None)
    logits = getattr(outputs, "logits", None)
    if pred_boxes is None or logits is None:
        latency_ms = (time.time() - t0) * 1000
        return PredictResponse(
            predictions=[[Prediction(label=0, class_name="(no detections)",
                                       probability=None, score=None)]],
            model_type=model_type, latency_ms=round(latency_ms, 2),
        )
    # logits: (1, num_queries, num_classes+1) — argmax over classes
    scores = torch.softmax(logits[0], dim=-1)
    cls_scores, cls_labels = scores[..., :-1].max(dim=-1)   # drop the no-object class
    boxes = pred_boxes[0]   # (num_queries, 4) cxcywh in [0,1]
    # Top boxes by confidence
    top_n = min(10, cls_scores.size(0))
    top_v, top_i = cls_scores.topk(top_n)
    preds: List[Prediction] = []
    for rank in range(top_n):
        i = int(top_i[rank].item())
        cx, cy, w, h = (float(v.item()) for v in boxes[i])
        x1, y1, x2, y2 = cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2
        label_id = int(cls_labels[i].item())
        name = class_names_fn(label_id) or f"class_{label_id}"
        preds.append(Prediction(
            label=label_id,
            class_name=f"{name} @ {x1:.2f},{y1:.2f},{x2:.2f},{y2:.2f}",
            probability=round(float(top_v[rank].item()), 6),
            score=None,
        ))
    latency_ms = (time.time() - t0) * 1000
    return PredictResponse(
        predictions=[preds],
        model_type=model_type,
        latency_ms=round(latency_ms, 2),
    )


def _depth_response(outputs, model_type: str, t0: float) -> "PredictResponse":
    """Render depth-estimation output as a single Prediction whose
    ``class_name`` summarizes the depth map's stats. The full map can be
    huge, so we don't ship it back over the wire — clients that want the
    raw tensor should call /predict via a dedicated endpoint."""
    depth = getattr(outputs, "predicted_depth", None)
    if depth is None:
        latency_ms = (time.time() - t0) * 1000
        return PredictResponse(
            predictions=[[Prediction(label=0, class_name="(no depth output)",
                                      probability=None, score=None)]],
            model_type=model_type, latency_ms=round(latency_ms, 2),
        )
    d = depth.float()
    summary = (
        f"depth map {tuple(d.shape)} — "
        f"min={float(d.min()):.3f} max={float(d.max()):.3f} "
        f"mean={float(d.mean()):.3f}"
    )
    latency_ms = (time.time() - t0) * 1000
    return PredictResponse(
        predictions=[[Prediction(label=0, class_name=summary,
                                  probability=None,
                                  score=round(float(d.mean().item()), 6))]],
        model_type=model_type,
        latency_ms=round(latency_ms, 2),
    )


def _masks_response(outputs, model_type: str, t0: float) -> "PredictResponse":
    """Render segmentation output as a summary string. The mask tensor
    itself is too large for a JSON response — clients that need pixel
    masks should consume the model directly."""
    masks = (getattr(outputs, "pred_masks", None)
             or getattr(outputs, "logits", None))
    latency_ms = (time.time() - t0) * 1000
    if masks is None:
        return PredictResponse(
            predictions=[[Prediction(label=0, class_name="(no mask output)",
                                      probability=None, score=None)]],
            model_type=model_type, latency_ms=round(latency_ms, 2),
        )
    return PredictResponse(
        predictions=[[Prediction(label=0,
                                  class_name=f"segmentation map {tuple(masks.shape)}",
                                  probability=None, score=None)]],
        model_type=model_type, latency_ms=round(latency_ms, 2),
    )
