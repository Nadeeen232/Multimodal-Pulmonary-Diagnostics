"""Audit Git-visible release files and measure a reproducible source ZIP.

Does not initialize the project's Git repository, commit, or push anything.
Outputs are written to the ignored .tmp/ folder.
"""

import hashlib
import json
import re
import subprocess
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

ROOT = Path(__file__).resolve().parents[1]
SCRATCH = ROOT / ".tmp"


def git_files():
    if (ROOT / ".git").exists():
        command = ["git"]
    else:
        audit_repo = SCRATCH / "release-audit.git"
        if not audit_repo.exists():
            subprocess.run(
                ["git", "init", "--bare", str(audit_repo)], check=True, stdout=subprocess.DEVNULL
            )
        command = ["git", f"--git-dir={audit_repo}", f"--work-tree={ROOT}"]
    raw = subprocess.check_output(
        [*command, "ls-files", "--cached", "--others", "--exclude-standard", "-z"], cwd=ROOT
    )
    return command, sorted(set(filter(None, raw.decode("utf-8").split("\0"))))


def main():
    SCRATCH.mkdir(exist_ok=True)
    command, files = git_files()
    probes = [
        "data/probe.csv",
        "data/README.md",
        "mlruns/probe.json",
        "optuna_study.db",
        "model.pt",
        "nested/model.pth",
        "embeddings.npy",
        "__pycache__/probe.pyc",
        ".env",
        "dist/probe.zip",
        ".venv/probe.txt",
        "docs/presentation.pptx",
        "docs/~$report.docx",
    ]
    result = subprocess.run(
        [*command, "check-ignore", "--no-index", "--stdin"],
        input=("\n".join(probes) + "\n").encode(),
        capture_output=True,
        cwd=ROOT,
    )
    ignored = set(result.stdout.decode().splitlines())
    if ignored != set(probes):
        raise ValueError("Required ignore rules failed: " + str(set(probes) - ignored))
    for report_name in (
        "docs/report.docx",
        "reports/report.docx",
        "reports/generated/check/report.docx",
    ):
        check = subprocess.run(
            [*command, "check-ignore", "--no-index", report_name],
            cwd=ROOT,
            capture_output=True,
        )
        if check.returncode != 1:
            raise ValueError("Normal Word reports must remain eligible: " + report_name)
    path_pattern = re.compile(r"(?<![A-Za-z])[A-Za-z]:[\\/]|/(?:Users|home)/[^/\s]+/")
    secret_pattern = re.compile(
        r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----|"
        r"\bgh[pousr]_[A-Za-z0-9]{30,}|\bAKIA[A-Z0-9]{16}\b"
    )
    issues, records, pending_deletions = [], [], []
    for name in files:
        path = ROOT / name
        if not path.is_file():
            pending_deletions.append(name)
            continue
        if (
            name.startswith(("data/", "mlruns/", ".venv/", ".tmp/", "dist/"))
            or path.suffix
            in {
                ".pt",
                ".pth",
                ".npy",
                ".pptx",
            }
            or path.name.startswith(("~$", ".~lock."))
        ):
            issues.append(f"Forbidden release asset: {name}")
        data = path.read_bytes()
        if len(data) >= 100 * 1024 * 1024:
            issues.append(f"File exceeds the GitHub 100 MiB limit: {name}")
        records.append(
            {"path": name, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
        )
        texts = []
        if path.suffix.lower() == ".docx":
            with ZipFile(path) as document:
                texts.extend(
                    (f"{name}!{part}", document.read(part).decode("utf-8-sig"))
                    for part in document.namelist()
                    if part.endswith((".xml", ".rels"))
                )
        elif path.suffix.lower() in {
            ".py",
            ".yaml",
            ".yml",
            ".json",
            ".jsonl",
            ".md",
            ".txt",
            ".xml",
            ".csv",
            ".toml",
            ".ps1",
            ".svg",
            ".html",
            ".ini",
            ".cfg",
        } or path.name in {
            "Dockerfile",
            "Dockerfile.update",
            "LICENSE",
            ".env.example",
            ".gitignore",
            ".dockerignore",
        }:
            # PowerShell can write UTF-16 evidence; scan its actual text too.
            content = data.decode(
                "utf-16" if data.startswith((b"\xff\xfe", b"\xfe\xff")) else "utf-8-sig"
            )
            texts.append((name, content))
        for text_name, content in texts:
            for index, line in enumerate(content.splitlines(), 1):
                if path_pattern.search(line):
                    issues.append(f"Absolute host path: {text_name}:{index}")
                if secret_pattern.search(line):
                    issues.append(f"Possible credential: {text_name}:{index}")
    if issues:
        raise ValueError("Release audit failed:\n" + "\n".join(issues))
    output = SCRATCH / "public_release_preview.zip"
    with ZipFile(output, "w", compression=ZIP_DEFLATED, compresslevel=9) as archive:
        for row in records:
            info = ZipInfo(row["path"], date_time=(2026, 1, 1, 0, 0, 0))
            info.compress_type = ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, (ROOT / row["path"]).read_bytes(), compresslevel=9)
    report = {
        "file_count": len(records),
        "source_payload_bytes": sum(row["bytes"] for row in records),
        "zip_bytes": output.stat().st_size,
        "zip_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "pending_deletions": pending_deletions,
        "largest_files": sorted(records, key=lambda row: row["bytes"], reverse=True)[:5],
        "size_definition": "Git-visible working-file bytes; excludes .git and ignored assets. Not a Git pack-size prediction.",
        "files": records,
    }
    (SCRATCH / "public_release_audit.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    (SCRATCH / "public_release_files.txt").write_text(
        "\n".join(row["path"] for row in records) + "\n", encoding="utf-8"
    )
    print(json.dumps({key: value for key, value in report.items() if key != "files"}, indent=2))


if __name__ == "__main__":
    main()
