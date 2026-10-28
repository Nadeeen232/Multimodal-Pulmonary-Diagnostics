"""Indiana pairing audit and deterministic report/patient group splits."""

from pathlib import Path

import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split

from pulmonary.common import file_hash, save_json
from pulmonary.images import PREPROCESS_VERSION, prepare_pil
from pulmonary.labels import LABEL_VERSION, model_text, weak_label


def link_duplicate_groups(frame):
    """Keep exact duplicate images and every associated study/patient in one split."""
    parent = {group: group for group in frame.group_id.unique()}

    def find(group):
        while parent[group] != group:
            parent[group] = parent[parent[group]]
            group = parent[group]
        return group

    duplicates = frame.loc[frame.image_sha256.ne("") & frame.image_sha256.duplicated(keep=False)]
    for _, rows in duplicates.groupby("image_sha256"):
        groups = sorted(set(rows.group_id))
        for group in groups[1:]:
            a, b = find(groups[0]), find(group)
            parent[max(a, b)] = min(a, b)
    result = frame.copy()
    result["group_id"] = result.group_id.map(lambda value: find(value))
    return result, len(duplicates)


def prepare(config, metadata_only=False):
    root = Path(config["data_root"]).resolve()
    files = {}
    for name in ("indiana_reports.csv", "indiana_projections.csv"):
        matches = list(root.rglob(name))
        if len(matches) != 1:
            raise FileNotFoundError(
                f"Expected exactly one {name} below {root}; found {len(matches)}"
            )
        files[name] = matches[0]
    reports = pd.read_csv(files["indiana_reports.csv"], dtype=str, keep_default_na=False)
    projections = pd.read_csv(files["indiana_projections.csv"], dtype=str, keep_default_na=False)
    for frame in (reports, projections):
        frame.columns = frame.columns.str.strip().str.lower()
    if not {"uid", "problems", "mesh"}.issubset(reports.columns):
        raise ValueError("Reports require uid, problems, mesh")
    if not {"uid", "filename"}.issubset(projections.columns):
        raise ValueError("Projections require uid and filename")
    audit = {
        "raw_reports": len(reports),
        "raw_projections": len(projections),
        "csv_sha256": {n: file_hash(p) for n, p in files.items()},
    }
    audit["duplicate_report_rows"] = int(reports.duplicated().sum())
    audit["duplicate_projection_rows"] = int(projections.duplicated().sum())
    reports, projections = reports.drop_duplicates().copy(), projections.drop_duplicates().copy()
    if reports.uid.eq("").any() or projections.uid.eq("").any():
        raise ValueError("Blank UID is not allowed")
    if reports.uid.duplicated().any():
        raise ValueError("Conflicting report rows share a UID; resolve before proceeding")
    if projections.filename.duplicated().any():
        raise ValueError("Repeated image filenames need manual pairing review")
    reports["medical_status"] = [
        weak_label(p, m) for p, m in zip(reports.problems, reports.mesh, strict=True)
    ]
    reports["model_text"] = [
        model_text(row, config["text_fields"], config["strip_target_terms"])
        for row in reports.to_dict("records")
    ]
    if reports.model_text.str.strip().eq("").all():
        raise ValueError("All permitted text is missing")
    patient_col = next(
        (col for col in ("patient_id", "patientid", "subject_id") if col in reports), None
    )
    if patient_col and reports[patient_col].eq("").any():
        raise ValueError(
            f"Incomplete patient identifier {patient_col}; resolve grouping explicitly"
        )
    reports["group_id"] = reports[patient_col] if patient_col else reports.uid
    audit["grouping_column"] = patient_col or "uid (report-level; patient independence unverified)"
    audit["report_label_counts"] = reports.medical_status.value_counts().to_dict()
    audit["missingness"] = {c: int(reports[c].eq("").sum()) for c in reports.columns}
    merged = projections.merge(
        reports, on="uid", how="left", validate="many_to_one", indicator=True
    )
    audit["unmatched_projection_rows"] = int(merged._merge.eq("left_only").sum())
    merged = merged.loc[merged._merge.eq("both")].drop(columns="_merge").copy()
    index = {}
    for path in root.rglob("*"):
        if path.suffix.lower() in {".png", ".jpg", ".jpeg"}:
            index.setdefault(path.name, []).append(path)
    paths, hashes, errors, cached, dimensions = [], [], [], [], []
    cache_root = Path(config["output_root"]) / "image_cache"
    if not metadata_only:
        cache_root.mkdir(parents=True, exist_ok=True)
    for name in merged.filename:
        if metadata_only:
            paths.append("")
            hashes.append("")
            cached.append("")
            continue
        matches = index.get(Path(name).name, [])
        try:
            if len(matches) != 1:
                raise ValueError(f"{len(matches)} image matches")
            with Image.open(matches[0]) as image:
                image.verify()
            digest = file_hash(matches[0])
            cache_path = cache_root / f"{digest}_{config['image_size']}.png"
            with Image.open(matches[0]) as image:
                dimensions.append({"width": image.width, "height": image.height})
                if not cache_path.exists():
                    temporary = cache_path.with_suffix(".tmp")
                    prepare_pil(image, config["image_size"]).save(temporary, format="PNG")
                    temporary.replace(cache_path)
            paths.append(matches[0].relative_to(root).as_posix())
            hashes.append(digest)
            cached.append(cache_path.relative_to(config["output_root"]).as_posix())
        except (OSError, ValueError) as exc:
            paths.append("")
            hashes.append("")
            cached.append("")
            errors.append({"filename": name, "error": str(exc)})
    merged["image_path"], merged["image_sha256"] = paths, hashes
    merged["processed_image_path"] = cached
    merged["sample_id"] = merged.uid + ":" + merged.filename
    out = Path(config["output_root"]) / "splits"
    out.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out / "metadata.csv", index=False)
    merged.loc[merged.medical_status.eq("uncertain")].to_csv(out / "uncertain.csv", index=False)
    eligible = merged.medical_status.ne("uncertain")
    if not metadata_only:
        eligible &= merged.image_path.ne("")
    frame = merged.loc[eligible].copy()
    if frame.empty:
        raise ValueError(
            "No labeled readable images found. Check data_root or use prepare-metadata for text-only work."
        )
    frame["target"] = frame.medical_status.map({"normal": 0, "abnormal": 1}).astype(int)
    frame, duplicate_rows = link_duplicate_groups(frame)
    audit["exact_duplicate_image_rows_linked"] = duplicate_rows
    audit["preprocessing_version"] = PREPROCESS_VERSION
    audit["image_size"] = config["image_size"]
    if dimensions:
        audit["image_dimensions_summary"] = pd.DataFrame(dimensions).describe().to_dict()
    # Patient groups may contain both targets. Stratify on group-level abnormal presence.
    groups = frame.groupby("group_id").target.max()
    train_ids, temp_ids = train_test_split(
        groups.index.to_numpy(),
        test_size=0.30,
        stratify=groups.to_numpy(),
        random_state=config["seed"],
    )
    val_ids, test_ids = train_test_split(
        temp_ids,
        test_size=0.50,
        stratify=groups.loc[temp_ids].to_numpy(),
        random_state=config["seed"],
    )
    splits = {"train": train_ids, "validation": val_ids, "test": test_ids}
    seen_groups, seen_hashes = set(), set()
    audit["splits"] = {}
    for name, ids in splits.items():
        part = frame.loc[frame.group_id.isin(ids)].sort_values("sample_id").reset_index(drop=True)
        if set(part.target) != {0, 1}:
            raise ValueError(f"Split {name} lacks a class")
        current_groups, current_hashes = set(part.group_id), set(part.image_sha256) - {""}
        if current_groups & seen_groups or current_hashes & seen_hashes:
            raise ValueError(
                "Group or identical-image leakage across splits; review duplicated studies"
            )
        seen_groups.update(current_groups)
        seen_hashes.update(current_hashes)
        part.to_csv(out / f"{name}.csv", index=False)
        audit["splits"][name] = {
            "rows": len(part),
            "groups": len(ids),
            "labels": {str(k): int(v) for k, v in part.target.value_counts().items()},
            "sha256": file_hash(out / f"{name}.csv"),
        }
    audit.update(
        {
            "label_version": LABEL_VERSION,
            "image_errors": errors,
            "excluded_uncertain_images": int(merged.medical_status.eq("uncertain").sum()),
            "text_fields": config["text_fields"],
            "strip_target_terms": config["strip_target_terms"],
            "seed": config["seed"],
        }
    )
    audit["image_validation"] = "pending-metadata-only" if metadata_only else "complete"
    from pulmonary.eda import write_eda

    write_eda(
        reports,
        frame.loc[frame.group_id.isin(train_ids)],
        audit,
        Path(config["reports_root"]) / "data",
        config["seed"],
    )
    save_json(Path(config["reports_root"]) / "data_audit.json", audit)
    save_json(out / "manifest.json", audit)
    return audit
