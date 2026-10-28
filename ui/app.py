"""Streamlit client; all model execution belongs to the FastAPI service.

The service is reached over HTTP (API_URL), or run in-process when
PULMONARY_IN_PROCESS=1 for single-container hosts such as Streamlit Community Cloud.
"""

import base64
import hashlib
import html
import os

import requests
import streamlit as st

st.set_page_config(page_title="Pulmonary Review Assistant", page_icon="🫁", layout="wide")
st.title("Pulmonary Review Assistant")
st.caption("Samsung Innovation Campus · Multimodal research prototype")
st.warning("Clinician verification is required. Model outputs are not clinically validated.")


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

with st.sidebar:
    st.subheader("Analysis settings")
    show_explanation = st.checkbox("Include explanations", value=True)
    st.caption("The operating threshold is fixed by validation analysis.")
    try:
        response = service.get("/health")
        if response.status_code >= 400:
            raise ValueError("Health check failed")
        health = response.json()
        if health["status"] == "ready":
            st.success("Model ready")
            st.caption(health["model_version"])
        else:
            st.info("The model is not ready. Training artifacts are required.")
    except (requests.RequestException, ValueError):
        st.error("The prediction service is unavailable.")

left, right = st.columns([1, 1])
with left:
    st.subheader("Radiograph and clinical context")
    upload = st.file_uploader(
        "Chest radiograph",
        type=["png", "jpg", "jpeg"],
        help="PNG or JPEG, up to 10 MiB. DICOM is not supported.",
    )
    if upload is not None:
        st.image(upload, caption="Uploaded image", width="stretch")
    indication = st.text_area(
        "Clinical indication",
        placeholder="Enter the associated pre-examination clinical indication…",
        max_chars=4000,
        height=140,
    )
    comparison = st.text_area(
        "Comparison context (optional)",
        max_chars=4000,
        help="Used only if the deployed model was trained with this field.",
    )
    analyze = st.button(
        "Analyze radiograph and text",
        type="primary",
        disabled=upload is None or not indication.strip(),
        width="stretch",
    )

fingerprint = hashlib.sha256(
    (upload.getvalue() if upload else b"") + indication.encode() + b"\x00" + comparison.encode()
).hexdigest()
if st.session_state.get("analysis_input") != fingerprint:
    st.session_state.pop("analysis", None)

if analyze:
    # Remove an old result before a new request so an error cannot leave a stale prediction.
    st.session_state.pop("analysis", None)
    with st.spinner("Analyzing both modalities…"):
        try:
            if upload.size > 10 * 1024 * 1024:
                raise ValueError("The image must be no larger than 10 MiB.")
            response = service.post(
                "/explain" if show_explanation else "/predict",
                files={"image": (upload.name, upload.getvalue(), upload.type)},
                data={"indication": indication, "comparison": comparison},
            )
            if response.status_code >= 400:
                try:
                    detail = response.json().get("detail", "Request failed")
                except ValueError:
                    detail = "Request failed"
                st.error(f"Service returned {response.status_code}: {detail}")
            else:
                st.session_state["analysis"] = response.json()
                st.session_state["analysis_input"] = fingerprint
        except requests.Timeout:
            st.error("Analysis timed out. Try again when the service is available.")
        except (requests.RequestException, ValueError) as exc:
            st.error(str(exc))

with right:
    st.subheader("Review result")
    result = st.session_state.get("analysis")
    if result:
        prediction = result.get("prediction", result)
        st.metric("Abnormality model score", f"{prediction['probability']:.1%}")
        st.progress(prediction["probability"])
        st.caption(
            "Uncalibrated score against weak annotation labels; not the probability of disease."
        )
        st.write(f"**Predicted Class:** {result['predicted_class']}")
        st.info(
            f"{prediction['priority_tier'].replace('_', ' ').title()} — {prediction['priority_reason']}"
        )
        st.caption(
            f"Decision threshold: {prediction['threshold']:.3f} · {prediction['model_version']}"
        )
        if "gradcam_base64" in result:
            st.image(
                base64.b64decode(result["gradcam_base64"]),
                caption="Grad-CAM of the deployed multimodal score",
                width="stretch",
            )
            st.write("**Text contributions**")
            for field, raw in [("indication", indication), ("comparison", comparison)]:
                highlights = sorted(
                    (item for item in result["text_highlights"] if item["field"] == field),
                    key=lambda item: item["start"],
                )
                if not highlights:
                    continue
                parts, cursor = [], 0
                for item in highlights:
                    parts.append(html.escape(raw[cursor : item["start"]]))
                    color = "#fee2e2" if item["contribution"] > 0 else "#dbeafe"
                    parts.append(
                        f'<span style="background:{color};color:#111827;padding:2px 3px;border-radius:3px">{html.escape(raw[item["start"] : item["end"]])}</span>'
                    )
                    cursor = item["end"]
                parts.append(html.escape(raw[cursor:]))
                st.markdown("".join(parts), unsafe_allow_html=True)
            st.caption(
                "Red: presence raises the score. Blue: presence lowers or does not change it. Contributions are context-dependent."
            )
            st.dataframe(result["text_highlights"][:15], hide_index=True, width="stretch")
            st.caption(result["method"])
            if result["text_tokens_truncated"]:
                st.caption("Text attribution was limited to the first 40 tokens.")
        st.subheader("Clinical report")
        st.text_area(
            "Rule-based report for clinician review",
            value=result["generated_report"],
            height=340,
            disabled=True,
        )
        st.download_button(
            "Download report text",
            data=result["generated_report"],
            file_name="clinical_report.txt",
            mime="text/plain",
        )
        st.download_button(
            "Download result JSON",
            data=__import__("json").dumps(result, indent=2),
            file_name="analysis.json",
            mime="application/json",
        )
    else:
        st.info("Upload a radiograph and enter its clinical indication to begin.")

with st.expander("How to interpret this prototype"):
    st.write(
        "The model learns a proxy abnormality task from Indiana structured annotations. The indication is not a full admission note. Image heatmaps and text score changes describe model sensitivity; they do not confirm diagnoses. Review priority is an illustrative queue ordering based on validation assumptions. A lower score does not rule out disease."
    )
