"""Streamlit client; all model execution belongs to the FastAPI service.

The service is reached over HTTP (API_URL), or run in-process when
PULMONARY_IN_PROCESS=1 for single-container hosts such as Streamlit Community Cloud.
Styling lives in theme.py and .streamlit/config.toml.
"""

import base64
import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path

import requests
import streamlit as st

HERE = Path(__file__).resolve().parent


def _load_theme():
    # Loaded by path: adding ui/ to sys.path would let this app.py shadow the app package.
    spec = importlib.util.spec_from_file_location("pulmonary_ui_theme", HERE / "theme.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


theme = _load_theme()

MAX_UPLOAD_BYTES = 10 * 1024 * 1024
DISCLAIMER = "Clinician verification is required. Model outputs are not clinically validated."
SAMPLE_LABELS = {
    "normal": "Normal case",
    "abnormal": "Abnormal case",
    "false_prediction": "Known miss",
}

st.set_page_config(
    page_title="Pulmonary Review Assistant", page_icon=":material/pulmonology:", layout="wide"
)


class HttpService:
    def __init__(self, url):
        self.url = url.rstrip("/")

    def get(self, path):
        return requests.get(self.url + path, timeout=5)

    def post(self, path, **kwargs):
        return requests.post(self.url + path, timeout=120, **kwargs)


@st.cache_resource(show_spinner="Loading the model…")
def in_process_service():
    from app.inprocess import InProcessService

    return InProcessService()


service = (
    in_process_service()
    if os.environ.get("PULMONARY_IN_PROCESS") == "1"
    else HttpService(os.environ.get("API_URL", "http://127.0.0.1:8080"))
)


def service_state():
    try:
        response = service.get("/health")
        if response.status_code >= 400:
            raise ValueError("Health check failed")
        health = response.json()
        if health["status"] == "ready":
            return "ready", health["model_version"]
        return "loading", None
    except (requests.RequestException, ValueError, KeyError):
        return "down", None


@st.cache_data
def samples():
    """Held-out test studies shipped with the repo (repo checkout or the UI image)."""
    for root in (HERE.parent / "demo_images", HERE / "demo_images"):
        table = root / "team29" / "demo_cases.csv"
        if table.is_file():
            with table.open(encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            return [
                {
                    "key": row["category"],
                    "label": SAMPLE_LABELS.get(row["category"], row["category"]),
                    "uid": row["uid"],
                    "weak_label": row["weak_label"],
                    "name": Path(row["filename"]).name,
                    "bytes": (root / row["filename"]).read_bytes(),
                    "indication": row["indication"],
                    "comparison": row["comparison"],
                }
                for row in rows
                if (root / row["filename"]).is_file()
            ]
    return []


def load_sample(sample):
    st.session_state.update(
        sample=sample["key"],
        indication=sample["indication"],
        comparison=sample["comparison"],
        upload_nonce=st.session_state.get("upload_nonce", 0) + 1,  # clears the uploader
    )


def clear_sample():
    st.session_state["sample"] = None


state, model_version = service_state()
theme_type = getattr(getattr(st.context, "theme", None), "type", None)
st.markdown(theme.css(theme_type), unsafe_allow_html=True)
st.html(theme.THEME_SCRIPT, unsafe_allow_javascript=True)
st.markdown(theme.header(state, model_version), unsafe_allow_html=True)

left, right = st.columns([0.45, 0.55], gap="large")
with left:
    st.markdown('<div class="pr-title">Study</div>', unsafe_allow_html=True)
    upload = st.file_uploader(
        "Chest radiograph",
        type=["png", "jpg", "jpeg"],
        help="PNG or JPEG, up to 10 MB. DICOM is not supported.",
        key=f"upload-{st.session_state.get('upload_nonce', 0)}",
        on_change=clear_sample,
    )
    cases = samples()
    if cases:
        st.markdown(
            '<div class="pr-hint">No X-ray at hand? Load a held-out test case:</div>',
            unsafe_allow_html=True,
        )
        for column, case in zip(st.columns(len(cases)), cases, strict=True):
            column.button(
                case["label"],
                key=f"sample-{case['key']}",
                on_click=load_sample,
                args=(case,),
                width="stretch",
            )

    sample = next((c for c in cases if c["key"] == st.session_state.get("sample")), None)
    if upload is not None:
        image = {"name": upload.name, "bytes": upload.getvalue(), "type": upload.type}
    elif sample is not None:
        image = {"name": sample["name"], "bytes": sample["bytes"], "type": "image/png"}
        st.caption(
            f"Test study {sample['uid']}, labeled {sample['weak_label']} in the dataset. "
            "The model never saw it during training."
        )
    else:
        image = None
    if image is not None:
        st.image(image["bytes"], width="stretch")

    indication = st.text_area(
        "Clinical indication",
        key="indication",
        placeholder="For example: shortness of breath for three days",
        max_chars=4000,
        height=110,
    )
    with st.expander(
        "Add comparison (optional)", expanded=bool(st.session_state.get("comparison"))
    ):
        comparison = st.text_area(
            "Comparison context",
            key="comparison",
            max_chars=4000,
            help="Used only if the deployed model was trained with this field.",
        )
    show_explanation = st.toggle(
        "Include explanations",
        value=True,
        help="Grad-CAM heatmap and text influence. Adds a few seconds.",
    )
    analyze = st.button(
        "Analyze",
        type="primary",
        disabled=image is None or not indication.strip() or state != "ready",
        width="stretch",
    )

fingerprint = hashlib.sha256(
    (image["bytes"] if image else b"") + indication.encode() + b"\x00" + comparison.encode()
).hexdigest()
if st.session_state.get("analysis_input") != fingerprint:
    st.session_state.pop("analysis", None)
    st.session_state.pop("analysis_error", None)

with right:
    st.markdown('<div class="pr-title">Result</div>', unsafe_allow_html=True)
    if analyze:
        # Remove an old result before a new request so an error cannot leave a stale prediction.
        st.session_state.pop("analysis", None)
        st.session_state.pop("analysis_error", None)
        st.session_state["analysis_input"] = fingerprint
        pending = st.empty()
        pending.markdown(theme.skeleton(), unsafe_allow_html=True)
        try:
            if len(image["bytes"]) > MAX_UPLOAD_BYTES:
                raise ValueError("The image must be no larger than 10 MiB.")
            response = service.post(
                "/explain" if show_explanation else "/predict",
                files={"image": (image["name"], image["bytes"], image["type"])},
                data={"indication": indication, "comparison": comparison},
            )
            if response.status_code >= 400:
                try:
                    detail = response.json().get("detail", "Request failed")
                except ValueError:
                    detail = "Request failed"
                st.session_state["analysis_error"] = (
                    f"Service returned {response.status_code}: {detail}"
                )
            else:
                st.session_state["analysis"] = response.json()
        except requests.Timeout:
            st.session_state["analysis_error"] = (
                "Analysis timed out. Try again when the service is available."
            )
        except (requests.RequestException, ValueError) as exc:
            st.session_state["analysis_error"] = str(exc)
        pending.empty()

    result = st.session_state.get("analysis")
    failure = st.session_state.get("analysis_error")
    if failure:
        st.markdown(theme.error(failure), unsafe_allow_html=True)
    elif result:
        prediction = result.get("prediction", result)
        st.markdown(theme.verdict(prediction, result["predicted_class"]), unsafe_allow_html=True)
        st.caption(
            "Uncalibrated score against weak annotation labels, not the probability of disease. "
            f"Model {prediction['model_version']}."
        )
        if "gradcam_base64" in result:
            heatmap = base64.b64decode(result["gradcam_base64"])
            view = st.segmented_control(
                "Image view",
                ["X-ray", "Heatmap", "Side by side"],
                default="Heatmap",
                key="viewer",
                label_visibility="collapsed",
            )
            if view == "X-ray":
                st.image(image["bytes"], width="stretch")
            elif view == "Side by side":
                original, overlay = st.columns(2)
                original.image(image["bytes"], caption="X-ray", width="stretch")
                overlay.image(heatmap, caption="Grad-CAM", width="stretch")
            else:
                st.image(heatmap, caption="Grad-CAM of the ResNet50 branch", width="stretch")

            st.markdown('<div class="pr-title">Text influence</div>', unsafe_allow_html=True)
            for field, raw in [("indication", indication), ("comparison", comparison)]:
                highlights = [item for item in result["text_highlights"] if item["field"] == field]
                if highlights:
                    st.markdown(theme.chips(raw, highlights), unsafe_allow_html=True)
            st.markdown(theme.LEGEND, unsafe_allow_html=True)
            with st.expander("Token details"):
                st.dataframe(result["text_highlights"][:15], hide_index=True, width="stretch")
                st.caption(result["method"])
                if result["text_tokens_truncated"]:
                    st.caption("Text attribution was limited to the first 40 tokens.")

        st.markdown('<div class="pr-title">Clinical report</div>', unsafe_allow_html=True)
        with st.container(border=True):
            st.markdown(theme.report(result["generated_report"]), unsafe_allow_html=True)
        save_text, save_json = st.columns(2)
        save_text.download_button(
            "Download report",
            data=result["generated_report"],
            file_name="clinical_report.txt",
            mime="text/plain",
            width="stretch",
        )
        save_json.download_button(
            "Download JSON",
            data=json.dumps(result, indent=2),
            file_name="analysis.json",
            mime="application/json",
            width="stretch",
        )
        with st.expander("Report as plain text"):
            st.code(result["generated_report"], language=None, wrap_lines=True)
    elif state == "ready":
        st.markdown(
            theme.empty(
                "No study analyzed yet",
                "Add a chest X-ray and its clinical indication, then select Analyze.",
            ),
            unsafe_allow_html=True,
        )
    elif state == "loading":
        st.markdown(
            theme.empty(
                "The model is not ready",
                "The model bundle has not loaded. Training artifacts are required.",
                warn=True,
            ),
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            theme.empty(
                "The prediction service is unavailable",
                "The app cannot reach the model service right now. Try again shortly.",
                warn=True,
            ),
            unsafe_allow_html=True,
        )

st.markdown(theme.footer(), unsafe_allow_html=True)
st.caption(DISCLAIMER)
with st.expander("How to read these results"):
    st.write(
        "The model learns a proxy abnormality task from Indiana structured annotations. The "
        "indication is not a full admission note. Image heatmaps and text score changes describe "
        "model sensitivity; they do not confirm diagnoses. Review priority is an illustrative "
        "queue ordering based on validation assumptions. A lower score does not rule out disease."
    )
