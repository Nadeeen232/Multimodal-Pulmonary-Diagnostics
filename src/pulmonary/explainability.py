"""Grad-CAM through the deployed fusion path and raw-note token occlusion."""

import base64
import io
import re

import matplotlib
import numpy as np
import torch
from PIL import Image

from pulmonary.images import image_tensor, prepare_pil
from report_generator import generate_report


def _image_tensor(engine, image):
    if hasattr(engine, "image_tensor"):
        return engine.image_tensor(image)
    return image_tensor(image, engine.manifest["image_size"])


def describe_region(cam):
    """Describe attribution in image coordinates; no anatomical segmentation implied."""
    cam = np.asarray(cam, dtype=float)
    if cam.ndim != 2 or not np.isfinite(cam).all() or cam.max() <= 0:
        return "unavailable (no positive Grad-CAM activation)"
    rows = np.array([part.sum() for part in np.array_split(cam, 3, axis=0)])
    cols = np.array([part.sum() for part in np.array_split(cam, 3, axis=1)])
    vertical = ["upper", "middle", "lower"][rows.argmax()]
    horizontal = ["image-left", "central", "image-right"][cols.argmax()]
    if rows.max() / rows.sum() < 0.5:
        return "diffuse across the radiograph (no dominant vertical image region)"
    if cols.max() / cols.sum() < 0.5:
        return f"{vertical} image region, spanning the image width"
    return f"{vertical} {horizontal} image region (patient laterality unverified)"


def gradcam(engine, image, indication, comparison="", return_region=False):
    activations = []
    handle = engine.encoder.features.register_forward_hook(
        lambda _m, _i, output: activations.append(output)
    )
    try:
        tensor = _image_tensor(engine, image).unsqueeze(0).requires_grad_(True)
        engine.encoder.zero_grad(set_to_none=True)
        score = engine.score_tensor(tensor, engine.text(indication, comparison))[0]
        activation = activations[0]
        gradient = torch.autograd.grad(score, activation)[0]
        weights = gradient.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((activation * weights).sum(dim=1))[0].detach().numpy()
        cam = cam / cam.max() if cam.max() > 0 else np.zeros_like(cam)
        size = engine.manifest["image_size"]
        mask = Image.fromarray(np.uint8(cam * 255)).resize((size, size), Image.Resampling.BILINEAR)
        colors = matplotlib.colormaps["inferno"](np.asarray(mask) / 255)[:, :, :3]
        prepared = (
            engine.prepare_image(image)
            if hasattr(engine, "prepare_image")
            else prepare_pil(image, size)
        )
        base = np.asarray(prepared).astype(float) / 255
        overlay = Image.fromarray(np.uint8(np.clip(0.65 * base + 0.35 * colors, 0, 1) * 255))
        output = io.BytesIO()
        overlay.save(output, format="PNG")
        encoded = base64.b64encode(output.getvalue()).decode("ascii")
        return (encoded, describe_region(cam)) if return_region else encoded
    finally:
        handle.remove()
        engine.encoder.zero_grad(set_to_none=True)


def explain(engine, image, indication, comparison="", max_tokens=40):
    # Serialize hooks and autograd with prediction calls to avoid shared-module races.
    with engine.lock:
        prediction = engine.predict(image, indication, comparison)
        overlay, region = gradcam(engine, image, indication, comparison, return_region=True)
        with torch.inference_mode():
            tensor = _image_tensor(engine, image).unsqueeze(0)
            context = engine.image_context(tensor)
        highlights = []
        examined = 0
        for field, raw in [("indication", indication), ("comparison", comparison)]:
            if field not in engine.manifest["text_fields"]:
                continue
            for match in re.finditer(r"\b[\w'-]+\b", raw):
                if examined >= max_tokens:
                    break
                altered = raw[: match.start()] + " " + raw[match.end() :]
                args = {"indication": indication, "comparison": comparison, field: altered}
                with torch.inference_mode():
                    p = float(engine.score_context(context, engine.text(**args))[0])
                highlights.append(
                    {
                        "field": field,
                        "token": match.group(),
                        "start": match.start(),
                        "end": match.end(),
                        "contribution": prediction["probability"] - p,
                    }
                )
                examined += 1
        highlights.sort(key=lambda value: abs(value["contribution"]), reverse=True)
        report_tokens = [
            item
            for item in highlights
            if abs(item["contribution"]) > 1e-8 and item["token"].lower() != "xxxx"
        ][:5]
        return {
            "prediction": prediction,
            "predicted_class": (
                "Normal" if prediction["predicted_label"] == "proxy_negative" else "Abnormal"
            ),
            "gradcam_base64": overlay,
            "text_highlights": highlights,
            "gradcam_region": region,
            "generated_report": generate_report(
                prediction["probability"], prediction["priority_tier"], region, report_tokens
            ),
            "method": "Grad-CAM of "
            + engine.manifest.get("gradcam_target", "final abnormal score")
            + "; raw-token leave-one-out occlusion conditioned on the image. Contributions are score changes, not additive causal effects.",
            "text_tokens_examined": examined,
            "text_tokens_truncated": sum(
                len(re.findall(r"\b[\w'-]+\b", raw))
                for field, raw in [("indication", indication), ("comparison", comparison)]
                if field in engine.manifest["text_fields"]
            )
            > examined,
        }
