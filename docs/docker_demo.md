# Docker demonstration (PowerShell)

Run these commands from a PowerShell terminal with Docker Desktop running:

```powershell
# Open a terminal at the repository root first.
docker compose up -d --wait --wait-timeout 180
docker compose ps
Invoke-RestMethod http://127.0.0.1:8080/ready
Invoke-RestMethod http://127.0.0.1:8080/model-info
Start-Process http://127.0.0.1:8080/docs
Start-Process http://127.0.0.1:8501
```

Both services should show `healthy`. These commands reuse the verified images.
On a clean machine, run `docker compose build` first. The first build downloads dependencies;
subsequent starts reuse the built images. API inference runs on CPU inside Docker.
The current frozen model bundle is mounted read-only from `artifacts/team_bundle`.

When rebuilding code on this machine, the optional dependency cache avoids
downloading the already installed libraries again:

```powershell
docker build -f Dockerfile.update -t sic_graduation_project-api:latest .
docker compose up -d --wait --wait-timeout 180
```

Use this code-only update with the locally built, trusted API image when
dependencies have not changed. It adds a small application layer. A clean
dependency build uses `docker compose build` and requires substantially more disk
space. `MODEL_BUNDLE_HOST` can override the Compose bundle path for a deliberate
rollback; the original v1 bundle remains at `artifacts/bundle`.

Demonstrate an actual image/text prediction:

```powershell
curl.exe --fail-with-body -X POST http://127.0.0.1:8080/predict -F "image=@demo_images/01_true_negative_uid_1330.png" -F "indication=XXXX-year-old male with shortness of breath."
```

In Streamlit, upload the same image, paste the same indication, leave explanations
enabled and click **Analyze radiograph and text**. Try the abnormal and false-positive
examples too; the matching indications are in `demo_images/*_indication.txt`.
Show the generated report below the results and download its text. It has Predicted
Class, Impression, Visual Findings, Clinical Context and Recommendation sections.
It is rule-based and states that clinician verification is required.
These cases were selected for demonstration and do not represent a random performance
estimate. `XXXX` is original deidentified text.

Show container logs and the internal UI-to-API network connection:

```powershell
docker compose logs --tail 30 api ui
docker compose exec -T ui python -c "import requests; print(requests.get('http://api:8080/ready', timeout=10).json())"
```

Repeat the automated readiness, real prediction, Grad-CAM, invalid-input and
in-container UI rendering verification (requires this project's local environment):

```powershell
.\.venv\Scripts\python.exe scripts/verify_docker_stack.py
```

Evidence is saved to `reports/generated/functional/docker_stack_verification.json`.
Stop the demo when finished:

```powershell
docker compose down
```
