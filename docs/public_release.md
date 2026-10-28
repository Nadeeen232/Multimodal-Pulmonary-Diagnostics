# Public GitHub release

The working project already has local Git history; nothing has been pushed by
this cleanup. Publish the clean export below into a new repository. Earlier local
commits contain a PowerPoint, which `.gitignore` cannot remove from history.

## Prepared release rules

`.gitignore` excludes the complete `data/` tree, local environments and caches,
MLflow runs, Optuna/SQLite databases, `.env` files, model checkpoints (`*.pt`,
`*.pth`), embedding arrays (`*.npy`, `*.npz`) and serialized models. Local process
records, `dist/`, all PowerPoint files and the original third-party presentation
template are also excluded. The deleted `team_model_review` documents have been
consolidated into `docs/technical_evidence_report.docx` and its Markdown source.
Selected reports, screenshots, documentation, demo images and generated deliverable
documents remain included.

Publishable source and text evidence were checked for absolute workstation paths.
Browser automation uses Playwright channels rather than an installed executable
path. Public Docker evidence omits host filesystem labels; the saved pytest output
uses a relative dependency path. These edits preserve measured results. Original
private experiment records remain in ignored local folders.

`LICENSE` supplies the requested MIT license for project code. The dataset,
third-party logos, template artwork and other upstream assets retain their own
terms; the MIT file does not relicense them. The source presentation template is
not distributed. The presentation builder uses that local template when present; a fresh checkout
uses a matching blank canvas. An explicit template can be passed with
`scripts/build_presentation.py --template <template-file>`.

## Audit and measure the exact release contents

```powershell
.\.venv\Scripts\python.exe scripts/audit_public_release.py
```

The audit uses Git's ignore logic, checks the release file list for prohibited
assets, scans text for workstation paths and selected credential patterns, and
creates two ignored local outputs:

- `.tmp/public_release_audit.json`: file names, exact byte sizes and SHA-256 hashes.
- `.tmp/public_release_preview.zip`: a deterministic ZIP of the current Git-visible files, without Git history.
- `.tmp/public_release_files.txt`: the complete list of files to publish.

`source_payload_bytes` is the exact sum of the released working-file sizes.
`zip_bytes` is the exact size of that ZIP. Neither is a prediction of `.git` or
GitHub's storage size: Git normalization, object compression and future history
affect that. Rerun the audit after making changes or staging files; it includes
tracked files too and will reject a prohibited asset accidentally tracked earlier.

## Initialize and push

On GitHub, create an **empty** repository named `pulmonary-review-assistant` under
your personal account. Do not add a README, license or `.gitignore` there because
this project already supplies them. Copy its HTTPS URL.

Run from this project's root in PowerShell. These commands extract the audited
source into a **new** folder and start fresh history there. They preserve the
original workspace and its Git history. Do not run `git push` from the original
workspace if the old presentation must remain private.

```powershell
.\.venv\Scripts\python.exe scripts/audit_public_release.py
$releaseFolder = Join-Path (Split-Path (Get-Location) -Parent) 'Pulmonary_Public_Release'
if (Test-Path -LiteralPath $releaseFolder) { throw 'Choose a new empty release folder.' }
Expand-Archive -LiteralPath .tmp/public_release_preview.zip -DestinationPath $releaseFolder
Set-Location -LiteralPath $releaseFolder
git init -b main
git add .
git status --short
git diff --cached --stat
git commit -m "Initial release: multimodal pulmonary review assistant"
git remote add origin https://github.com/YOUR_GITHUB_USERNAME/pulmonary-review-assistant.git
git push -u origin main
```

Replace `YOUR_GITHUB_USERNAME` with your account name. Git may request your author
identity on the first commit; configure your actual name and GitHub-associated
email, then repeat the commit. Authenticate through your credential manager or
GitHub CLI when prompted; do not embed a token in the remote URL or repository.

## Running on another computer

Follow the README setup instructions and restore a trusted model bundle or train
one. Ignored datasets and model files are deliberately absent from a Git clone.
Docker uses the same relative bundle mount on any host supporting Linux containers.

On macOS/Linux, the Python environment equivalents are:

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install torch==2.13.0 torchvision==0.28.0 --index-url https://download.pytorch.org/whl/cpu
.venv/bin/python -m pip install -c constraints-cpu.txt -e '.[ui,data,dev]'
```

Select the appropriate Torch installation for your operating system and processor
if the CPU index does not provide that platform's wheel. Browser automation defaults
to installed Edge on Windows and Playwright Chromium elsewhere. To use Chromium:

```bash
.venv/bin/python -m pip install playwright
.venv/bin/python -m playwright install chromium
.venv/bin/python scripts/automated_demo.py --browser chromium
```

The optional Docker Desktop window-capture utility is Windows-specific; it is not
part of training or serving. All screenshots already included in the release can
be viewed without running that utility.

## Current submission handoff

Release 0.2.0-team29-47e281fc uses `artifacts/team_bundle`. The verified offline
model archive is `dist/SIC_0.2.0-team29_model_bundle.zip`, intentionally excluded
from Git. Provide it separately alongside the source submission and restore it
before serving a fresh checkout. Its checksum is recorded in the submission
manifest. Word/Markdown evidence and the recorded demo are in `docs/`. PowerPoint
files are excluded from publication; generate the private submission deck using
`scripts/build_presentation.py` and deliver it separately.

The audit detects specified path and credential patterns, verifies ignore rules,
and inspects Word XML as well as text. It does not guarantee that every possible
secret or third-party licensing issue has been detected. Review included screenshots
and public demo assets before publishing. Model archives are intentionally absent
from the source export, so restore the trusted bundle before starting inference.
