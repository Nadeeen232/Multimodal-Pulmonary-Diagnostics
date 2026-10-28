"""Verify the running Compose stack and save actual container evidence."""

import base64
import csv
import json
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
API = "http://127.0.0.1:8080"


def get(url):
    with urllib.request.urlopen(url, timeout=30) as response:
        return response.status, response.read()


def main():
    raw_services = subprocess.check_output(
        ["docker", "compose", "ps", "--format", "json"], cwd=ROOT, text=True
    ).strip()
    services = (
        json.loads(raw_services)
        if raw_services.startswith("[")
        else [json.loads(line) for line in raw_services.splitlines() if line.strip()]
    )
    # Publish operational evidence without host filesystem paths in Docker labels.
    services = [
        {
            key: service[key]
            for key in ["Service", "Name", "ID", "Image", "State", "Health", "Ports"]
        }
        for service in services
    ]
    assert {s["Service"] for s in services} == {"api", "ui"}, services
    assert all(s["State"] == "running" and s["Health"] == "healthy" for s in services)
    evidence = {"status": "passed", "services": services, "checks": {}}
    for name, url in {
        "ready": f"{API}/ready",
        "docs": f"{API}/docs",
        "ui_health": "http://127.0.0.1:8501/_stcore/health",
        "ui_page": "http://127.0.0.1:8501/",
    }.items():
        status, body = get(url)
        assert status == 200
        evidence["checks"][name] = {"status_code": status}
        if name == "ready":
            evidence["model"] = json.loads(body)
    with (ROOT / "demo_images/demo_cases.csv").open(encoding="utf-8", newline="") as handle:
        row = next(csv.DictReader(handle))
    boundary = "sic-docker-verification-boundary"
    image = (ROOT / "demo_images" / row["filename"]).read_bytes()
    payload = (
        (
            f'--{boundary}\r\nContent-Disposition: form-data; name="indication"\r\n\r\n'
            f"{row['indication']}\r\n--{boundary}\r\n"
            'Content-Disposition: form-data; name="comparison"\r\n\r\n'
            f"{row.get('comparison', '')}\r\n--{boundary}\r\n"
            f'Content-Disposition: form-data; name="image"; filename="demo.png"\r\n'
            "Content-Type: image/png\r\n\r\n"
        ).encode()
        + image
        + f"\r\n--{boundary}--\r\n".encode()
    )
    folder = ROOT / "reports/generated/functional"
    folder.mkdir(parents=True, exist_ok=True)
    for route in ["predict", "explain"]:
        started = time.perf_counter()
        request = urllib.request.Request(
            f"{API}/{route}",
            data=payload,
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        )
        with urllib.request.urlopen(request, timeout=120) as response:
            result = json.load(response)
            assert response.status == 200
        prediction = result.get("prediction", result)
        assert all(
            section in result["generated_report"]
            for section in [
                "Predicted Class:",
                "Impression:",
                "Visual Findings:",
                "Clinical Context:",
                "Recommendation:",
            ]
        )
        assert "educational" not in result["generated_report"].lower()
        assert "Clinician verification is required" in result["generated_report"]
        assert result["predicted_class"] == (
            "Normal" if row["predicted_label"] == "proxy_negative" else "Abnormal"
        )
        assert (
            "image" in result["gradcam_region"]
            or "unavailable" in result["gradcam_region"]
            or "radiograph" in result["gradcam_region"]
        )
        assert prediction["model_version"] == evidence["model"]["model_version"]
        assert prediction["predicted_label"] == row["predicted_label"]
        assert abs(prediction["probability"] - float(row["probability"])) < 1e-4
        if route == "explain":
            assert base64.b64decode(result["gradcam_base64"]).startswith(b"\x89PNG")
        evidence["checks"][route] = {
            "status_code": 200,
            "latency_seconds": time.perf_counter() - started,
            "uid": row["uid"],
            "prediction": prediction,
            "generated_report": result["generated_report"],
            "gradcam_region": result["gradcam_region"],
        }
        (folder / f"docker_{route}_response.json").write_text(json.dumps(result, indent=2))
    try:
        urllib.request.urlopen(
            urllib.request.Request(
                f"{API}/predict",
                data=b"indication=cough",
                headers={"Content-Type": "application/x-www-form-urlencoded"},
            ),
            timeout=30,
        )
        raise AssertionError("Missing image was accepted")
    except urllib.error.HTTPError as error:
        assert error.code == 422
        evidence["checks"]["missing_image"] = {"status_code": error.code}
    code = (
        "import os, sys, json, hashlib; from streamlit.testing.v1 import AppTest; "
        "assert os.environ['API_URL'] == 'http://api:8080'; "
        "result=json.load(sys.stdin); "
        "page=AppTest.from_file('/ui/app.py', default_timeout=60); "
        "page.session_state['analysis']=result; "
        "page.session_state['analysis_input']=hashlib.sha256(bytes([0])).hexdigest(); "
        "page.run(); "
        "assert not page.exception, str(page.exception); "
        "assert any(m.value == 'Model ready' for m in page.success); "
        "assert all('educational' not in w.value.lower() for w in page.warning); "
        "assert any(t.value == result['generated_report'] for t in page.text_area); "
        "assert page.button[0].disabled; print('Container UI rendered; internal API ready')"
    )
    rendered = subprocess.run(
        ["docker", "compose", "exec", "-T", "ui", "python", "-c", code],
        cwd=ROOT,
        text=True,
        timeout=120,
        input=json.dumps(result),
        stdout=subprocess.PIPE,
        check=True,
    ).stdout
    evidence["checks"]["ui_render_inside_container"] = {
        "status": "passed",
        "output": rendered.strip(),
    }
    output = folder / "docker_stack_verification.json"
    output.write_text(json.dumps(evidence, indent=2), encoding="utf-8")
    print(json.dumps(evidence, indent=2))


if __name__ == "__main__":
    main()
