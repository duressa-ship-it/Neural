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
from typing import Any, AsyncIterator, Dict, List, Optional, Union

import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel as PydanticModel, Field

from neural_platform.core.config import ExperimentConfig


# ---------------------------------------------------------------------------
# Request / Response schemas (with rich examples for the /docs page)
# ---------------------------------------------------------------------------

class ChatContentPart(PydanticModel):
    """One typed slot inside a ChatMessage — text, an image, or audio.

    Multimodal HF chat templates (Gemma-3, Qwen2-VL, LLaVA, Idefics)
    accept a list of these per message; the modern HF processors know
    how to render the typed-parts list directly into the model's
    expected token / pixel sequence.
    """
    type:       str = Field(..., description="'text' | 'image' | 'audio'")
    text:       Optional[str] = Field(None, description="When type='text'.")
    image_b64:  Optional[str] = Field(None, description="When type='image' — base64 PNG/JPG.")
    audio_b64:  Optional[str] = Field(None, description="When type='audio' — base64 WAV/MP3/etc.")


class ChatMessage(PydanticModel):
    """One turn of conversation. ``content`` is either a plain string
    (single-text message — works with every chat template) or a list of
    typed parts for multimodal turns."""
    role:    str = Field(..., description="'system' | 'user' | 'assistant'")
    content: Union[str, List[ChatContentPart]] = Field(
        ...,
        description=(
            "Plain string for text-only turns, or a list of ChatContentParts "
            "for multimodal (text + image + audio). The server runs "
            "processor.apply_chat_template(messages, …) to render."
        ),
    )


class PredictRequest(PydanticModel):
    """
    Flexible prediction request — supply *one* of these fields based on
    your model type:

    * **inputs**:    flat list of floats (MLP, RNN), or a list of timesteps
    * **tokens**:    list of token IDs (transformer, when bypassing the tokenizer)
    * **image_b64**: base64-encoded image bytes (CNN)
    * **text**:      raw string (transformer with HuggingFace tokenizer)
    * **messages**:  list of ChatMessage for multi-turn / chat-template
                     models (Gemma-3, Qwen2-VL, LLaVA, …). When present,
                     overrides text/image_b64.

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
    audio_b64: Optional[str] = Field(
        None,
        description=(
            "Base64-encoded audio file (WAV/FLAC/MP3/OGG/M4A). The server "
            "decodes it to a float32 mono waveform and resamples to the "
            "model's expected rate (Whisper: 16 kHz). Use instead of the "
            "raw `inputs` waveform when sending real audio files from the UI."
        ),
    )
    video_b64: Optional[str] = Field(
        None,
        description=(
            "Base64-encoded video file (MP4/AVI/MOV). The server decodes it "
            "to a frame tensor at the model's expected resolution. Falls "
            "back to a frame-extraction error if torchvision/av aren't "
            "available."
        ),
    )
    file_mime: Optional[str] = Field(
        None,
        description=(
            "Optional MIME hint for any of the *_b64 fields. Used to pick "
            "the right decoder when the file extension is ambiguous (e.g. "
            "raw audio vs. video container). The server will still sniff "
            "magic bytes when this is missing."
        ),
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
    messages: Optional[List[ChatMessage]] = Field(
        None,
        description=(
            "Multi-turn chat history. When present and the loaded model "
            "exposes a chat template, the server runs "
            "`processor.apply_chat_template(messages, add_generation_prompt=True)` "
            "and uses the result as the model's input — superseding the "
            "single-turn `text` / `image_b64` fields."
        ),
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
    """One prediction. ``class_name`` is populated when the checkpoint
    includes class labels.

    ``metadata`` carries structured-output details that don't fit in a
    flat (label, probability) pair:

      * **Object detection** — ``{"bbox": [x1, y1, x2, y2], "format": "xyxy_norm"}``
        (boxes are in [0,1] of the image dimensions).
      * **QA spans** — ``{"start_idx": int, "end_idx": int}`` so the
        client can highlight the matched span in the input.
      * **Token classification** — ``{"offsets": [[s,e],…], "tokens": [...]}``
        the per-token labels with character offsets back into the user's text.

    The frontend dispatches on :attr:`PredictResponse.result_kind`; this
    field is the data backing each renderer.
    """
    label: int = Field(..., description="Class index (0-based)")
    class_name: Optional[str] = Field(None, description="Human-readable class name, if known")
    probability: Optional[float] = Field(None, description="Softmax probability (0–1)")
    score: Optional[float] = Field(None, description="Raw logit value")
    metadata: Optional[Dict[str, Any]] = Field(
        None,
        description=(
            "Structured-output details for non-classification tasks "
            "(bbox / qa span indices / token offsets / depth stats / mask "
            "thumbnails). Schema depends on PredictResponse.result_kind."
        ),
    )


class PredictResponse(PydanticModel):
    """Top-K predictions for each input sample.

    ``result_kind`` lets the client pick the right renderer without
    inferring from the task name:

      * ``"logits"`` — standard classification (default; existing top-K bars).
      * ``"generated_text"`` — ASR / summarization / translation /
        text-generation / image-to-text. Single Prediction with the
        decoded text in ``class_name``.
      * ``"qa_spans"`` — question-answering. Single Prediction whose
        ``metadata`` has ``start_idx`` / ``end_idx`` in addition to the
        decoded answer string.
      * ``"boxes"`` — object detection. One Prediction per detected box,
        bbox in metadata.
      * ``"depth"`` — depth estimation. One Prediction whose metadata
        carries a base64 PNG colormap thumbnail of the depth map.
      * ``"masks"`` — segmentation. Top-N class masks, each as a base64
        PNG thumbnail in metadata.
      * ``"token_spans"`` — token classification. One Prediction per
        non-trivial span, with character offsets back into the input.
    """
    predictions: List[List[Prediction]] = Field(..., description="One inner list per sample")
    model_type: str
    latency_ms: float = Field(..., description="Server-side inference latency in ms")
    result_kind: str = Field(
        "logits",
        description="What kind of structured result this is — drives the UI's renderer choice.",
    )


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
    has_chat_template: bool = Field(
        False,
        description=(
            "True when the loaded tokenizer / processor exposes a "
            "chat_template — drives the Predict tab to surface the "
            "chat transcript pane instead of the single-turn input."
        ),
    )
    # UI hint dictionary derived from the pipeline_specs table — the
    # Predict tab reads this off /info and renders a single universal
    # input panel whose visible fields adapt to the connected model.
    ui_hint:          Optional[Dict[str, Any]] = Field(
        None,
        description=(
            "Hint object describing which input fields the universal "
            "Predict panel should expose for this model. Schema is stable "
            "across releases — see core.pipeline_specs.ui_hint."
        ),
    )


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
        ui_hint_dict = None
        if config.model.type.value == "hf_pipeline":
            try:
                from neural_platform.core.pipeline_specs import (
                    resolve as _resolve_spec, PROCESSOR_AUTO_CLASS, ui_hint,
                )
                spec = _resolve_spec(pipeline_task)
                auto_class = spec.auto_class
                processor_class = PROCESSOR_AUTO_CLASS.get(spec.processor_kind)
                ui_hint_dict = ui_hint(spec)
            except Exception:
                pass
        else:
            # Native model types (mlp/cnn/rnn/audio_cnn/…) — emit a sensible
            # default UI hint so the same universal panel renders for them.
            mtype = config.model.type.value
            ui_hint_dict = _native_ui_hint(mtype)

        # Detect chat-template support on the loaded processor /
        # tokenizer. The Predict UI surfaces a chat transcript pane
        # whenever this is True, regardless of pipeline_task.
        has_chat_template = False
        try:
            proc = container.get("processor")
            tok = getattr(proc, "tokenizer", proc)
            has_chat_template = bool(getattr(tok, "chat_template", None))
        except Exception:
            pass
        if ui_hint_dict is not None:
            ui_hint_dict = {**ui_hint_dict, "show_chat": has_chat_template}

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
            ui_hint=ui_hint_dict,
            has_chat_template=has_chat_template,
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

    @app.post("/predict/stream", tags=["Inference"],
              summary="Stream tokens from a generative model via SSE",
              responses={
                  400: {"description": "Task isn't generative — call /predict instead."},
                  422: {"description": "Input validation failed."},
                  503: {"description": "Model not loaded yet."},
              })
    async def predict_stream(request: PredictRequest):
        """Token-by-token streaming for generative tasks.

        Returns ``text/event-stream`` with two event types:

          * ``token`` — one per generated piece, ``data: <text>``.
          * ``done``  — terminator, ``data: {…}`` with final text +
            latency stats.

        Only fires when the loaded model's pipeline_task has
        ``needs_generation=True`` in the spec table. Other tasks 400
        with a clear message — caller should use the regular /predict.

        Long-running. Each server can stream concurrently up to its
        process budget; cancellation comes from the client closing
        the connection (StreamingResponse exits via the generator's
        GeneratorExit and the streamer is allowed to drain).
        """
        from neural_platform.core.pipeline_specs import resolve as _resolve_spec
        from fastapi.responses import StreamingResponse as _SSE
        spec = (_resolve_spec(config.training.pipeline_task)
                if config.model.type.value == "hf_pipeline"
                else _resolve_spec(None))
        if not spec.needs_generation:
            raise HTTPException(
                400,
                f"Task '{spec.task}' isn't generative — call /predict instead. "
                f"Streaming is available for: ASR, summarization, translation, "
                f"text-generation, image-to-text, image-text-to-text, any-to-any.",
            )
        model = container.get("model")
        if model is None:
            raise HTTPException(503, "Model not loaded")

        # Build the input on the request thread so any 422 surfaces
        # before we open the SSE connection.
        try:
            tensor_input = _build_input(
                request, config.model.type.value, device, config,
                container.get("processor"),
            )
        except _InputError as exc:
            raise HTTPException(422, str(exc))
        except Exception as exc:
            raise HTTPException(422, f"Input error: {exc}")

        async def event_source():
            async for chunk in _stream_generation(
                model=model, processor=container.get("processor"),
                tensor_input=tensor_input, request=request, spec=spec,
            ):
                yield chunk

        return _SSE(event_source(), media_type="text/event-stream", headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        })

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
                        result_kind="generated_text",
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

        # Token classification — per-token labels with character offsets
        # so the UI can highlight spans in the original text. Runs *before*
        # the generic logits path because the model output is logits but
        # we want span-level rendering, not document-level top-K.
        if spec.output_kind == POSTPROC_TOKEN_LOGITS:
            tensor = raw_output if isinstance(raw_output, torch.Tensor) else getattr(raw_output, "logits", None)
            if isinstance(tensor, torch.Tensor):
                return _token_spans_response(
                    tensor, tensor_input, container.get("processor"),
                    model_type, t0, class_names, request.top_k,
                )

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

    # Chat short-circuit: when the request carries `messages`, render
    # via the HF chat template regardless of the underlying model type.
    # Useful for transformer + hf_pipeline servers that the user wants
    # to talk to with the chat surface.
    if request.messages is not None and len(request.messages) > 0 and tokenizer is not None:
        return _build_chat_input(request, tokenizer, device)

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


def _build_chat_input(request: PredictRequest, processor, device):
    """Render a multi-turn ChatMessage list via the HF chat template.

    Modern HF tokenizers ship a Jinja chat template
    (``tokenizer.chat_template``) that knows how to format messages for
    the specific model — system / user / assistant roles, multimodal
    content parts, generation prompt, etc. We hand the messages to
    ``processor.apply_chat_template(messages, …)`` and let it produce
    the right input tensors.

    Falls back to a plain concatenation when no chat template is
    available — keeps simple text-generation servers usable with the
    chat surface even on older tokenizers.
    """
    if processor is None:
        raise _InputError(
            "Chat input requires a loaded processor / tokenizer. "
            "Restart with a model whose tokenizer is on the Hub."
        )

    tok = getattr(processor, "tokenizer", processor)
    has_template = bool(getattr(tok, "chat_template", None))

    # Detect multimodal content first — we only need to decode binary
    # blobs when we're actually going to feed them to the chat template.
    # Doing it eagerly would crash the no-template fallback on otherwise
    # valid (but multimodal) chat sessions before we even get to the
    # "this model can't accept multimodal chat" error.
    has_image = any(
        not isinstance(m.content, str) and any(
            p.type == "image" and p.image_b64 for p in m.content
        ) for m in request.messages
    )
    has_audio = any(
        not isinstance(m.content, str) and any(
            p.type == "audio" and p.audio_b64 for p in m.content
        ) for m in request.messages
    )

    # No-template + multimodal: surface the limitation now, before we
    # spend the cycles to decode the binary blobs.
    if not has_template and (has_image or has_audio):
        raise _InputError(
            "This model has no chat_template; can't accept multimodal chat "
            "messages without one. Either upgrade the tokenizer or send a "
            "single-turn request via `text` + `image_b64`."
        )

    # Now build the message list. For the chat-template path we decode
    # binary parts to PIL.Image / waveform; for the fallback we only
    # care about text.
    msgs = []
    for m in request.messages:
        if isinstance(m.content, str):
            msgs.append({"role": m.role, "content": m.content})
            continue
        parts = []
        for part in m.content:
            if part.type == "text" and part.text is not None:
                parts.append({"type": "text", "text": part.text})
            elif part.type == "image" and part.image_b64 is not None and has_template:
                # Chat templates (Qwen2-VL / LLaVA / Idefics) want a real
                # PIL.Image, not the b64 string. Decode lazily — only
                # touches PIL on the chat-template path.
                parts.append({"type": "image", "image": _decode_image_b64(part.image_b64)})
            elif part.type == "audio" and part.audio_b64 is not None and has_template:
                wav = _decode_audio_b64(
                    part.audio_b64,
                    target_sr=_resolve_target_sample_rate(processor),
                )
                parts.append({"type": "audio", "audio": wav})
        msgs.append({"role": m.role, "content": parts})

    # Path 1: native chat template (modern tokenizers / unified processors).
    if has_template:
        try:
            # Newer transformers prefer apply_chat_template returning
            # tensors directly (return_tensors='pt'); older returns a
            # plain string we'd then have to tokenize ourselves.
            try:
                enc = processor.apply_chat_template(
                    msgs, add_generation_prompt=True,
                    tokenize=True, return_tensors="pt", return_dict=True,
                )
            except (TypeError, AttributeError):
                rendered = tok.apply_chat_template(
                    msgs, add_generation_prompt=True, tokenize=False,
                )
                enc = tok(rendered, return_tensors="pt")
            return {k: (v.to(device) if hasattr(v, "to") else v)
                     for k, v in dict(enc).items()}
        except Exception as exc:
            raise _InputError(
                f"The chat template rejected the messages: {exc}. "
                "Try a single-turn `text` field, or trim message roles "
                "the model doesn't accept (some models reject 'system')."
            )

    # Path 2: no chat template — concatenate text parts as a fallback.
    # Loses the role / multimodal information but keeps simple chat
    # working with vanilla CausalLMs. Multimodal chat was already
    # rejected above, so by this point everything is text-only.
    flat = "\n".join(
        f"{m['role'].upper()}: {m['content']}"
        if isinstance(m["content"], str)
        else f"{m['role'].upper()}: " + " ".join(p.get("text", "") for p in m["content"])
        for m in msgs
    ) + "\nASSISTANT:"
    enc = tok(flat, return_tensors="pt", padding=True, truncation=True)
    return {k: v.to(device) for k, v in dict(enc).items()}


def _decode_image_b64(b64: str):
    """Tiny helper — decode a base64 image into a PIL.Image. Used by
    the chat path; the regular image input adapter inlines this."""
    import base64 as _b64
    import io as _io
    from PIL import Image
    return Image.open(_io.BytesIO(_b64.b64decode(b64))).convert("RGB")


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

    # ----- chat template short-circuit -----------------------------------
    # When the request carries a `messages` list AND the loaded
    # processor / tokenizer has a chat template, render via the modern
    # HF chat path. This wins over the per-task adapter below so a
    # generic `text-generation` server can host a chat session with no
    # extra config — the same Gemma-3 / Qwen2-VL / Llama-3 pattern as
    # the official transformers demos.
    if request.messages is not None and len(request.messages) > 0:
        return _build_chat_input(request, processor, device)

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
        if request.audio_b64 is not None:
            target_sr = _resolve_target_sample_rate(processor)
            proc_kwargs["audio"] = _decode_audio_b64(
                request.audio_b64, target_sr=target_sr,
                file_mime=request.file_mime,
            )
            proc_kwargs.setdefault("sampling_rate", target_sr)
        elif request.inputs is not None:
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
        # Accept three input shapes for universal-input UX:
        #   1. audio_b64 — a base64-encoded audio file (WAV/FLAC/MP3/…).
        #      The Predict tab's universal file drop produces this.
        #   2. inputs    — a flat list of float32 waveform samples.
        #   3. inputs    — nested 2D, in which case we take the first row.
        target_sr = _resolve_target_sample_rate(processor)
        if request.audio_b64 is not None:
            wav = _decode_audio_b64(
                request.audio_b64,
                target_sr=target_sr,
                file_mime=request.file_mime,
            )
        elif request.inputs is not None:
            data = request.inputs
            if isinstance(data, list) and data and isinstance(data[0], (int, float)):
                wav = [float(x) for x in data]
            elif isinstance(data, list) and data and isinstance(data[0], (list, tuple)):
                wav = [float(x) for x in data[0]]   # take first sample if 2D was sent
            else:
                wav = list(data)
        else:
            raise _InputError(
                f"Task '{spec.task}' needs an audio waveform. Send `audio_b64` "
                "(base64-encoded WAV/FLAC/MP3) or `inputs` (a flat list of float "
                f"samples at {target_sr} Hz)."
            )

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
        # Universal input: accept either a real video file via
        # `video_b64` or the legacy 4D nested-list `inputs` shape.
        if request.video_b64 is not None:
            frames = _decode_video_b64(
                request.video_b64, file_mime=request.file_mime,
            )
            # frames: (T, C, H, W) → (1, T, C, H, W) batched.
            return frames.unsqueeze(0).to(device)
        if request.inputs is None:
            raise _InputError(
                f"Task '{spec.task}' needs a video. Send `video_b64` (base64-"
                "encoded MP4/MOV) or `inputs` (a 4D nested list (T, C, H, W))."
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
# Binary file decoders (used by the universal predict input)
# ---------------------------------------------------------------------------
# The Predict UI sends one universal request shape (text + context +
# image_b64 + audio_b64 + video_b64). The decoders below turn those
# base64 blobs into the tensors / arrays the model's processor needs.
# Centralizing decode here keeps the input adapter thin and lets us
# share the audio resampling logic between INPUT_AUDIO and INPUT_ANY
# without duplication.

async def _stream_generation(*, model, processor, tensor_input, request,
                              spec) -> "AsyncIterator[str]":
    """Drive ``model.encoder.generate(streamer=…)`` on a background
    thread and yield SSE-formatted chunks.

    The generation call is blocking — it spins inside ``transformers``
    until either max_new_tokens or an EOS token is hit. To deliver
    tokens to the client as they're produced, we hand
    ``TextIteratorStreamer`` to ``.generate()``, run that on a thread,
    and pump the streamer from this async generator.

    Yields strings already wrapped in SSE form (``event: …\\ndata: …\\n\\n``)
    so the FastAPI ``StreamingResponse`` can emit them directly.
    """
    import asyncio
    import json as _json
    import threading
    import time as _time
    try:
        from transformers import TextIteratorStreamer
    except ImportError:
        yield "event: error\ndata: " + _json.dumps({
            "detail": "transformers TextIteratorStreamer unavailable. "
                       "Upgrade transformers or call /predict instead.",
        }) + "\n\n"
        return

    encoder = getattr(model, "encoder", model)
    if not hasattr(encoder, "generate"):
        yield "event: error\ndata: " + _json.dumps({
            "detail": f"Loaded model has no .generate() method — task "
                       f"'{spec.task}' isn't actually streamable here.",
        }) + "\n\n"
        return

    # Find the right tokenizer to feed the streamer. For unified
    # processors (Whisper, BLIP) this lives on .tokenizer; plain
    # AutoTokenizer instances are themselves the tokenizer.
    tokenizer = getattr(processor, "tokenizer", processor)
    if tokenizer is None:
        yield "event: error\ndata: " + _json.dumps({
            "detail": "No tokenizer loaded — can't stream decoded text.",
        }) + "\n\n"
        return

    streamer = TextIteratorStreamer(
        tokenizer, skip_prompt=True, skip_special_tokens=True,
    )
    gen_inputs = (tensor_input if isinstance(tensor_input, dict)
                  else {"inputs": tensor_input})

    gen_kwargs: Dict[str, Any] = {"streamer": streamer}
    if request.max_new_tokens is not None:
        gen_kwargs["max_new_tokens"] = int(request.max_new_tokens)
    if request.temperature is not None:
        gen_kwargs["temperature"] = float(request.temperature)
    if request.do_sample is not None:
        gen_kwargs["do_sample"] = bool(request.do_sample)
    # Sensible default cap so a buggy model can't run forever.
    gen_kwargs.setdefault("max_new_tokens", 256)

    # Run generate() on a thread — it's a blocking call, and we need
    # the event loop free to ferry chunks out.
    error_box: Dict[str, Any] = {}
    def _run():
        try:
            with torch.no_grad():
                encoder.generate(**gen_inputs, **gen_kwargs)
        except Exception as exc:
            error_box["error"] = exc
            # End the streamer so the consumer loop exits cleanly.
            try:
                streamer.end()
            except Exception:
                pass
    thread = threading.Thread(target=_run, daemon=True)
    thread.start()

    t0 = _time.time()
    pieces: List[str] = []
    loop = asyncio.get_event_loop()
    try:
        while True:
            # streamer is a sync iterator; pull next() in a thread so
            # we don't block the event loop.
            piece = await loop.run_in_executor(None, _next_or_sentinel,
                                                  streamer)
            if piece is _STREAM_END:
                break
            pieces.append(piece)
            yield f"event: token\ndata: {_json.dumps(piece)}\n\n"
        # generate() exited — wait briefly for the thread to settle so
        # we capture any post-generation errors.
        thread.join(timeout=0.5)
        if "error" in error_box:
            yield "event: error\ndata: " + _json.dumps({
                "detail": str(error_box["error"]),
            }) + "\n\n"
            return
        latency_ms = (_time.time() - t0) * 1000
        yield "event: done\ndata: " + _json.dumps({
            "text":         "".join(pieces),
            "latency_ms":   round(latency_ms, 2),
            "result_kind":  "generated_text",
        }) + "\n\n"
    finally:
        # If the client disconnected mid-stream, the generator gets
        # closed and we land here. Make sure the worker thread isn't
        # left hanging on a dead streamer.
        try:
            streamer.end()
        except Exception:
            pass


# Sentinel returned by the executor when the streamer iterator is
# exhausted. Plain StopIteration doesn't propagate well across
# run_in_executor boundaries.
_STREAM_END = object()


def _next_or_sentinel(streamer):
    """Pull one piece off a TextIteratorStreamer.

    Returns ``_STREAM_END`` when the iterator is exhausted. We avoid
    propagating ``StopIteration`` through the executor because asyncio
    treats it as a programming error rather than normal completion.
    """
    try:
        return next(streamer)
    except StopIteration:
        return _STREAM_END
    except Exception:
        return _STREAM_END


def _native_ui_hint(model_type: str) -> Dict[str, Any]:
    """UI hint for native (non-HF) model types.

    The Predict tab's universal panel reads this off /info just like the
    HF-pipeline hint, so users see a consistent layout whether they're
    serving a from-scratch CNN or a Hub-pulled Whisper. Each branch picks
    the right primary field + file accept type.
    """
    base: Dict[str, Any] = {
        "show_text":             True,
        "show_context":          False,
        "show_file":             False,
        "show_candidate_labels": False,
        "show_generation_knobs": False,
        "accept":                "",
        "text_placeholder":      "Input value",
        "file_placeholder":      "Drop a file here",
        "primary_field":         "text",
        "summary":               f"Model type: {model_type}",
    }
    if model_type == "mlp":
        base.update(
            show_text=False, show_file=False,
            primary_field="inputs",
            summary="Send a flat list of float features in `inputs`.",
        )
    elif model_type in ("cnn", "video_cnn"):
        base.update(
            show_text=False, show_file=True,
            accept="image/*" if model_type == "cnn" else "video/*",
            file_placeholder=("Drop an image" if model_type == "cnn"
                               else "Drop a video file"),
            primary_field="image_b64" if model_type == "cnn" else "video_b64",
        )
    elif model_type == "audio_cnn":
        base.update(
            show_text=False, show_file=True, accept="audio/*",
            file_placeholder="Drop an audio file",
            primary_field="audio_b64",
        )
    elif model_type == "rnn":
        base.update(
            show_text=True, primary_field="inputs",
            text_placeholder="Sequence — JSON 2D list, or comma-separated rows",
        )
    elif model_type == "transformer":
        base.update(
            text_placeholder="Type the text to classify / continue",
        )
    elif model_type in ("tcn", "tabular"):
        base.update(
            text_placeholder="Numeric features (JSON list or comma-separated)",
            primary_field="inputs",
        )
    return base


def _resolve_target_sample_rate(processor) -> int:
    """Pick the sample rate the loaded HF audio processor expects.

    Whisper / Wav2Vec2 / HuBERT all want 16 kHz; some music models want
    44.1 kHz. We read the processor's own ``sampling_rate`` attribute
    (works for both FeatureExtractor and the unified Processor that
    wraps one), and fall back to 16000 — the right answer for ~95% of
    HF audio models.
    """
    for src in (processor, getattr(processor, "feature_extractor", None)):
        if src is None:
            continue
        sr = getattr(src, "sampling_rate", None)
        if sr:
            try:
                return int(sr)
            except Exception:
                pass
    return 16000


def _decode_audio_b64(b64: str, *, target_sr: int = 16000,
                       file_mime: Optional[str] = None) -> "list[float]":
    """Decode a base64-encoded audio file to a mono float32 waveform.

    Tries soundfile first (covers WAV/FLAC/OGG natively), falls back to
    librosa (handles MP3/M4A via audioread/ffmpeg). Mono-mixes stereo and
    resamples to ``target_sr`` (the rate the loaded HF feature extractor
    expects — Whisper / Wav2Vec2 want 16 kHz).

    Returns a plain Python list of floats so it slots into the existing
    ``request.inputs`` path with no further conversion. Raises
    :class:`_InputError` with an actionable message when the file can't
    be decoded — keeps the Predict UI's error pane meaningful.
    """
    import base64
    import io
    try:
        raw = base64.b64decode(b64, validate=False)
    except Exception as exc:
        raise _InputError(f"Couldn't base64-decode audio_b64: {exc}")

    # Path 1: soundfile (no external dependencies; covers WAV/FLAC/OGG).
    try:
        import soundfile as sf
        import numpy as np
        wav, sr = sf.read(io.BytesIO(raw), dtype="float32", always_2d=False)
        if hasattr(wav, "ndim") and wav.ndim > 1:
            wav = wav.mean(axis=1)   # stereo → mono
        # Resample if needed. Prefer librosa (high quality) when present;
        # otherwise fall back to a simple linear interp via numpy.
        if sr != target_sr:
            try:
                import librosa
                wav = librosa.resample(wav, orig_sr=sr, target_sr=target_sr)
            except ImportError:
                # Linear interp — adequate for short ASR clips, far from
                # ideal for production audio. The librosa path is always
                # preferred when available.
                ratio = target_sr / sr
                new_len = int(round(len(wav) * ratio))
                xs_old = np.linspace(0, 1, len(wav), endpoint=False)
                xs_new = np.linspace(0, 1, new_len, endpoint=False)
                wav = np.interp(xs_new, xs_old, wav)
        return [float(x) for x in wav]
    except _InputError:
        raise
    except Exception:
        # Path 2: librosa (handles MP3/M4A via audioread/ffmpeg). librosa
        # is in the [audio] extra; if it's not installed, surface a clear
        # missing-dep error instead of an opaque traceback.
        try:
            import librosa
            wav, _ = librosa.load(io.BytesIO(raw), sr=target_sr, mono=True)
            return [float(x) for x in wav]
        except ImportError as exc:
            raise _InputError(
                f"Couldn't decode the audio file ({file_mime or 'unknown mime'}). "
                "Install the audio extra (`pip install -e .[audio]`) so soundfile "
                "and librosa are available, or send a raw waveform via the "
                "`inputs` field instead."
            ) from exc
        except Exception as exc:
            raise _InputError(
                f"Could not decode audio file ({file_mime or 'unknown'}): {exc}. "
                "Try a different format (WAV/FLAC are the most reliable)."
            ) from exc


def _decode_video_b64(b64: str, *, file_mime: Optional[str] = None,
                       num_frames: int = 16,
                       target_size: tuple = (224, 224)) -> "torch.Tensor":
    """Decode a base64-encoded video file to a (T, C, H, W) float tensor.

    Uses torchvision.io.read_video when available (handles MP4/MOV/AVI
    via ffmpeg). Uniformly samples ``num_frames`` frames across the clip
    and resizes them to ``target_size``. Returns a tensor in [0, 1] float
    space.

    The `[video]` extra installs torchvision + av. Without that, a clear
    install hint surfaces instead of an opaque traceback.
    """
    import base64
    import io
    import tempfile
    try:
        raw = base64.b64decode(b64, validate=False)
    except Exception as exc:
        raise _InputError(f"Couldn't base64-decode video_b64: {exc}")

    try:
        from torchvision.io import read_video
        from torchvision import transforms as T
    except ImportError as exc:
        raise _InputError(
            "Video decoding needs torchvision (and ffmpeg). Install with "
            "`pip install -e .[video]`."
        ) from exc

    # torchvision.io.read_video accepts a path, not bytes — write to a
    # temp file. Cleaned up on context exit.
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=True) as f:
        f.write(raw)
        f.flush()
        try:
            frames, _audio, _info = read_video(f.name, pts_unit="sec")
        except Exception as exc:
            raise _InputError(
                f"Could not decode video file ({file_mime or 'unknown'}): {exc}. "
                "MP4 is the most reliable format. Make sure ffmpeg is installed."
            ) from exc

    # frames: (T, H, W, C) uint8. Subsample uniformly to num_frames.
    if frames.shape[0] == 0:
        raise _InputError("Video decoded to zero frames — the file may be corrupt.")
    if frames.shape[0] > num_frames:
        idx = torch.linspace(0, frames.shape[0] - 1, num_frames).long()
        frames = frames[idx]
    elif frames.shape[0] < num_frames:
        # Pad by repeating the last frame. Better than dropping the
        # request — short clips become longest-clip * num_frames.
        last = frames[-1:].expand(num_frames - frames.shape[0], -1, -1, -1)
        frames = torch.cat([frames, last], dim=0)

    # (T, H, W, C) → (T, C, H, W) and float in [0, 1]
    frames = frames.permute(0, 3, 1, 2).float() / 255.0
    # Resize each frame to the target HxW.
    resize = T.Resize(target_size, antialias=True)
    frames = torch.stack([resize(f) for f in frames])
    return frames


# ---------------------------------------------------------------------------
# Structured-output postprocessors
# ---------------------------------------------------------------------------
# Each of these takes the bare HF output object that the wrapper passed
# through (start_logits + end_logits for QA, pred_boxes + scores for
# object detection, etc.) and turns it into a PredictResponse the Predict
# UI can render. Surface area is deliberately the same as the
# logits-based path: a single-sample top-K list with `class_name`
# carrying the human-readable answer.

def _tensor_to_png_b64(tensor: torch.Tensor, *, max_dim: int = 256) -> Optional[str]:
    """Render a 2-D float tensor as a base64-encoded PNG thumbnail.

    Used for depth maps + segmentation masks where the raw tensor is
    too large to ship over JSON but a small thumbnail is plenty for the
    Predict UI to display. Min-max normalizes to [0, 255], applies a
    perceptual colormap (viridis when matplotlib is around, plain
    grayscale otherwise), and downsamples to fit ``max_dim`` on the
    longer side.
    """
    try:
        import io
        import base64
        from PIL import Image
        import numpy as np

        t = tensor.detach().float().cpu()
        # Squeeze leading singletons until we have 2-D HxW.
        while t.dim() > 2 and t.size(0) == 1:
            t = t.squeeze(0)
        if t.dim() != 2:
            # Multi-channel / per-class — take argmax over channels for masks.
            if t.dim() == 3:
                t = t.argmax(dim=0).float()
            else:
                return None
        # Min-max normalize to [0, 1].
        lo, hi = float(t.min()), float(t.max())
        if hi - lo > 1e-9:
            t = (t - lo) / (hi - lo)
        else:
            t = torch.zeros_like(t)
        arr = (t.numpy() * 255).astype(np.uint8)

        # Optional viridis colormap when matplotlib is installed. mpl 3.7+
        # moved cm.get_cmap → matplotlib.colormaps; try the new API first
        # so we don't trip the deprecation warning on every depth render.
        try:
            try:
                import matplotlib   # type: ignore
                cmap = matplotlib.colormaps.get_cmap("viridis")
            except (ImportError, AttributeError):
                import matplotlib.cm as cm   # type: ignore
                cmap = cm.get_cmap("viridis")
            colored = cmap(arr / 255.0)
            arr = (colored[..., :3] * 255).astype(np.uint8)
            img = Image.fromarray(arr, mode="RGB")
        except Exception:
            img = Image.fromarray(arr, mode="L")

        # Thumbnail down to keep the response small.
        img.thumbnail((max_dim, max_dim))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("ascii")
    except Exception:
        return None


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
            metadata={
                "start_idx": int(start),
                "end_idx": int(end),
                "start_prob": round(s_prob, 6),
                "end_prob": round(e_prob, 6),
            },
        )]],
        model_type=model_type,
        latency_ms=round(latency_ms, 2),
        result_kind="qa_spans",
    )


def _boxes_response(outputs, model_type: str, t0: float, class_names_fn
                     ) -> "PredictResponse":
    """Render object-detection output as one Prediction per detected box.

    Each Prediction carries:
      * ``class_name`` — the human-readable label (no longer the
        coordinate-stuffed string the UI used to parse).
      * ``probability`` — detection score (softmax over class logits).
      * ``metadata.bbox`` — ``[x1, y1, x2, y2]`` in normalized [0, 1]
        image-space coordinates. ``metadata.format`` always
        ``"xyxy_norm"`` so the client knows how to scale.

    The frontend overlays these on the previewed image via a canvas;
    older clients still see the label + score with no special handling.
    """
    pred_boxes = getattr(outputs, "pred_boxes", None)
    logits = getattr(outputs, "logits", None)
    if pred_boxes is None or logits is None:
        latency_ms = (time.time() - t0) * 1000
        return PredictResponse(
            predictions=[[Prediction(label=0, class_name="(no detections)",
                                       probability=None, score=None)]],
            model_type=model_type, latency_ms=round(latency_ms, 2),
            result_kind="boxes",
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
            class_name=name,
            probability=round(float(top_v[rank].item()), 6),
            score=None,
            metadata={
                "bbox": [round(x1, 6), round(y1, 6),
                          round(x2, 6), round(y2, 6)],
                "format": "xyxy_norm",
            },
        ))
    latency_ms = (time.time() - t0) * 1000
    return PredictResponse(
        predictions=[preds],
        model_type=model_type,
        latency_ms=round(latency_ms, 2),
        result_kind="boxes",
    )


def _depth_response(outputs, model_type: str, t0: float) -> "PredictResponse":
    """Render depth-estimation output as a viridis colormap thumbnail.

    The depth map can be hundreds of KB as a raw tensor — far too big
    for a single Predict round-trip. Min-max-normalize, colormap, and
    ship a 256×256 PNG thumbnail in metadata.image_b64. The frontend
    renders that inline and shows the depth-stat summary alongside.
    """
    depth = getattr(outputs, "predicted_depth", None)
    latency_ms = (time.time() - t0) * 1000
    if depth is None:
        return PredictResponse(
            predictions=[[Prediction(label=0, class_name="(no depth output)",
                                      probability=None, score=None)]],
            model_type=model_type, latency_ms=round(latency_ms, 2),
            result_kind="depth",
        )
    d = depth.float()
    png_b64 = _tensor_to_png_b64(d)
    summary = (
        f"depth map {tuple(d.shape)} — "
        f"min={float(d.min()):.3f} max={float(d.max()):.3f} "
        f"mean={float(d.mean()):.3f}"
    )
    return PredictResponse(
        predictions=[[Prediction(
            label=0,
            class_name=summary,
            probability=None,
            score=round(float(d.mean().item()), 6),
            metadata={
                "image_b64":  png_b64,
                "image_mime": "image/png",
                "shape":      list(d.shape),
                "min":        round(float(d.min()), 6),
                "max":        round(float(d.max()), 6),
                "mean":       round(float(d.mean()), 6),
            },
        )]],
        model_type=model_type,
        latency_ms=round(latency_ms, 2),
        result_kind="depth",
    )


def _masks_response(outputs, model_type: str, t0: float) -> "PredictResponse":
    """Render segmentation output as one Prediction per non-trivial
    class mask, each carrying a base64 PNG thumbnail of that class's
    mask + a coverage percentage.

    For models that emit per-class logits ``(C, H, W)``, we argmax to
    a single label map and surface up to ``top_n`` distinct classes
    ranked by pixel coverage.
    """
    masks = (getattr(outputs, "pred_masks", None)
             or getattr(outputs, "logits", None))
    latency_ms = (time.time() - t0) * 1000
    if masks is None:
        return PredictResponse(
            predictions=[[Prediction(label=0, class_name="(no mask output)",
                                      probability=None, score=None)]],
            model_type=model_type, latency_ms=round(latency_ms, 2),
            result_kind="masks",
        )

    m = masks.float()
    # Squeeze leading singletons (1, C, H, W) → (C, H, W)
    while m.dim() > 3 and m.size(0) == 1:
        m = m.squeeze(0)

    preds: List[Prediction] = []
    if m.dim() == 3:
        # (C, H, W): argmax across classes, then surface top-N by area.
        label_map = m.argmax(dim=0)
        unique, counts = torch.unique(label_map, return_counts=True)
        # Sort by count descending; cap at 6 classes to keep payload small.
        order = torch.argsort(counts, descending=True)
        for rank in range(min(6, unique.size(0))):
            idx = int(order[rank].item())
            cls_id = int(unique[idx].item())
            count = int(counts[idx].item())
            coverage = count / float(label_map.numel())
            mask_bin = (label_map == cls_id).float()
            png_b64 = _tensor_to_png_b64(mask_bin)
            preds.append(Prediction(
                label=cls_id,
                class_name=f"class_{cls_id}",
                probability=round(coverage, 6),
                score=None,
                metadata={
                    "image_b64":  png_b64,
                    "image_mime": "image/png",
                    "coverage":   round(coverage, 6),
                    "shape":      list(label_map.shape),
                },
            ))
    else:
        # 2-D mask (rare): single Prediction with the whole map.
        preds.append(Prediction(
            label=0,
            class_name=f"segmentation map {tuple(m.shape)}",
            probability=None, score=None,
            metadata={
                "image_b64":  _tensor_to_png_b64(m),
                "image_mime": "image/png",
                "shape":      list(m.shape),
            },
        ))

    return PredictResponse(
        predictions=[preds],
        model_type=model_type, latency_ms=round(latency_ms, 2),
        result_kind="masks",
    )


def _token_spans_response(logits: torch.Tensor, tensor_input,
                            processor, model_type: str, t0: float,
                            class_names: Optional[List[str]],
                            top_k: int) -> "PredictResponse":
    """Render token-classification output as per-token labels with
    character offsets back into the input text.

    Used for NER, POS tagging, and other token-level tasks. The frontend
    highlights the spans in the user's text input. Falls back to the
    averaged top-K path (the existing _build_predictions behavior) when
    the tokenizer doesn't provide offsets.
    """
    latency_ms = (time.time() - t0) * 1000
    # logits: typically (1, seq_len, num_classes). Squeeze to (seq, C).
    while logits.dim() > 2 and logits.size(0) == 1:
        logits = logits.squeeze(0)
    if logits.dim() != 2:
        # Unexpected shape — fall back to the regular top-K path.
        return PredictResponse(
            predictions=[_build_predictions(logits, top_k, True, class_names)],
            model_type=model_type, latency_ms=round(latency_ms, 2),
            result_kind="logits",
        )

    probs = torch.softmax(logits, dim=-1)
    preds_per_token = probs.argmax(dim=-1)
    confs = probs.max(dim=-1).values

    # Try to recover character offsets from the tokenizer. The processor
    # may carry that on its `tokenizer` attribute (Whisper-style) or be
    # a tokenizer itself.
    offsets = None
    tokens = None
    input_ids = (tensor_input.get("input_ids")
                  if isinstance(tensor_input, dict) else None)
    tok = getattr(processor, "tokenizer", processor)
    if input_ids is not None and tok is not None:
        try:
            ids = (input_ids[0] if input_ids.dim() > 1 else input_ids).tolist()
            tokens = tok.convert_ids_to_tokens(ids)
        except Exception:
            tokens = None

    spans: List[Prediction] = []
    seq_len = preds_per_token.size(0)
    for i in range(seq_len):
        cls_id = int(preds_per_token[i].item())
        # Skip O / padding by default — the user-facing surface is the
        # *non-trivial* tagged tokens. We treat label 0 as the
        # background class (matches the BIO convention).
        if cls_id == 0:
            continue
        name = _lookup_class(class_names, cls_id) or f"class_{cls_id}"
        spans.append(Prediction(
            label=cls_id,
            class_name=name,
            probability=round(float(confs[i].item()), 6),
            score=None,
            metadata={
                "token_idx": i,
                "token":     (tokens[i] if tokens and i < len(tokens) else None),
            },
        ))
        if len(spans) >= top_k:
            break

    if not spans:
        # Fallback: at least show the document-level top-K so the UI
        # has something to render.
        return PredictResponse(
            predictions=[_build_predictions(
                logits, top_k, True, class_names)],
            model_type=model_type, latency_ms=round(latency_ms, 2),
            result_kind="logits",
        )

    return PredictResponse(
        predictions=[spans],
        model_type=model_type, latency_ms=round(latency_ms, 2),
        result_kind="token_spans",
    )
