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
    top_k: int = Field(5, ge=1, le=100, description="How many top predictions to return")
    return_probabilities: bool = Field(True, description="Include softmax probabilities")

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
    parameter_count: Optional[int]
    trainable_parameters: Optional[int]
    checkpoint_path: str
    device: str
    class_names: Optional[List[str]] = Field(
        None, description="Human-readable class labels indexed by class id"
    )
    output_size: Optional[int] = None
    epoch: Optional[int] = Field(None, description="Epoch the loaded checkpoint was saved at")
    val_loss: Optional[float] = Field(None, description="Best validation loss at checkpoint time")


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
        # Try to find a tokenizer for transformer text inputs
        container["tokenizer"] = _try_load_tokenizer(config)

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
        n_params = model.count_parameters(trainable_only=False) if hasattr(model, "count_parameters") else None
        n_trainable = model.count_parameters(trainable_only=True) if hasattr(model, "count_parameters") else None

        # Output size depends on the model family
        output_size = None
        try:
            output_size = config.model.get_arch_config().output_size
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
            class_names=meta.get("class_names"),
            output_size=output_size,
            epoch=meta.get("epoch"),
            val_loss=meta.get("val_loss"),
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

        t0 = time.time()
        model_type = config.model.type.value

        try:
            tensor_input = _build_input(request, model_type, device, config, container.get("tokenizer"))
        except _InputError as e:
            raise HTTPException(status_code=422, detail=str(e))
        except Exception as e:
            raise HTTPException(status_code=422, detail=f"Input error: {e}")

        with torch.no_grad():
            try:
                if isinstance(tensor_input, dict):
                    logits = model(**tensor_input)
                elif isinstance(tensor_input, (list, tuple)):
                    logits = model(*tensor_input)
                else:
                    logits = model(tensor_input)
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


def _try_load_tokenizer(config: ExperimentConfig):
    """
    Best-effort tokenizer load. Used for transformer + hf_pipeline text
    tasks so the client can send raw `text` instead of pre-tokenizing.
    """
    mtype = config.model.type.value
    if mtype not in ("transformer", "hf_pipeline"):
        return None
    try:
        from transformers import AutoTokenizer
    except ImportError:
        return None
    # Choose the tokenizer name from the model config first, then fall back
    # to the data transforms section, then a generic default.
    name = None
    if mtype == "transformer" and config.model.transformer and config.model.transformer.use_pretrained:
        name = config.model.transformer.use_pretrained
    if mtype == "hf_pipeline" and config.model.hf_pipeline and config.model.hf_pipeline.pretrained:
        name = config.model.hf_pipeline.pretrained
    if not name and config.data.transforms and isinstance(config.data.transforms, dict):
        name = config.data.transforms.get("text", {}).get("tokenizer")
    name = name or "bert-base-uncased"
    # Pass HF token through if we have one — gated tokenizers (e.g. Llama)
    # need it. Token never leaves this process; we read via hf_auth.
    kwargs = {}
    try:
        from neural_platform.core.hf_auth import get_token as _hf_token
        t = _hf_token()
        if t:
            kwargs["token"] = t
    except Exception:
        pass
    try:
        return AutoTokenizer.from_pretrained(name, **kwargs)
    except Exception:
        return None


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
                              device, tokenizer):
    """Dispatch on the configured `pipeline_task` to pick the right adapter.

    Text → tokenize via the loaded tokenizer (or use raw `tokens`).
    Image → decode `image_b64` and resize to whatever the model expects.
    Audio → accept `inputs` (waveform samples) — same shape as audio_cnn.
    Tabular / numeric → flat `inputs` vector.
    Anything we don't have a preprocessor for falls back to raw `inputs`.
    """
    task = (config.training.pipeline_task or "").lower().strip()
    text_tasks = {
        "text-classification", "token-classification", "question-answering",
        "summarization", "translation", "text-generation", "fill-mask",
        "feature-extraction", "sentence-similarity", "zero-shot-classification",
    }
    image_tasks = {
        "image-classification", "image-segmentation", "object-detection",
        "depth-estimation", "image-to-image", "image-to-text",
        "zero-shot-image-classification", "keypoint-detection",
    }
    audio_tasks = {
        "audio-classification", "automatic-speech-recognition",
        "voice-activity-detection", "audio-to-audio",
    }

    if task in text_tasks:
        if request.text is not None:
            if tokenizer is None:
                raise _InputError(
                    "No tokenizer is loaded on this server. Pre-tokenize on the "
                    "client and send 'tokens' instead, or restart the server with "
                    "a model whose tokenizer is on the Hub."
                )
            enc = tokenizer(
                request.text,
                max_length=128,
                padding="max_length", truncation=True, return_tensors="pt",
            )
            return {k: v.to(device) for k, v in enc.items()}
        if request.tokens is not None:
            tokens = request.tokens
            if not isinstance(tokens[0], (list, tuple)):
                tokens = [tokens]
            return {"input_ids": torch.tensor(tokens, dtype=torch.long).to(device)}
        raise _InputError(
            f"Task '{task}' needs 'text' (server-tokenized) or 'tokens' (pre-tokenized)."
        )

    if task in image_tasks:
        if request.image_b64 is None:
            raise _InputError(
                f"Task '{task}' needs 'image_b64' — base64-encoded PNG/JPG bytes."
            )
        import base64, io
        try:
            from PIL import Image
            import torchvision.transforms as T
        except ImportError as exc:
            raise _InputError(f"Image inference needs Pillow + torchvision: {exc}")
        img = Image.open(io.BytesIO(base64.b64decode(request.image_b64))).convert("RGB")
        # Generic 224×224 resize — most HF image models accept this.
        transform = T.Compose([T.Resize((224, 224)), T.ToTensor()])
        return transform(img).unsqueeze(0).to(device)

    if task in audio_tasks:
        if request.inputs is None:
            raise _InputError(
                f"Task '{task}' needs 'inputs' as a list of waveform samples "
                "(default sample rate 16000)."
            )
        data = request.inputs
        if isinstance(data, list) and data and isinstance(data[0], (int, float)):
            wav = torch.tensor([data], dtype=torch.float32)
        else:
            wav = torch.tensor(data, dtype=torch.float32)
        return wav.to(device)

    # Fallback: pass `inputs` straight through as a flat float tensor.
    if request.inputs is None:
        raise _InputError(
            f"Task '{task or 'custom'}' needs 'inputs' (list of floats) or one of "
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
    """Convert raw logits into a list of Prediction objects."""
    if logits.size(0) == 1:
        # Binary or single-output regression head
        prob = torch.sigmoid(logits).item()
        label = int(prob > 0.5)
        return [Prediction(
            label=label,
            class_name=_lookup_class(class_names, label),
            probability=round(prob, 6) if return_probs else None,
            score=round(float(logits[0].item()), 6),
        )]

    probs = torch.softmax(logits, dim=0)
    k = min(top_k, logits.size(0))
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
