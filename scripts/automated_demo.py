"""Run the real UID 1330 Streamlit demo in a fresh Playwright browser session.

Install: python -m pip install playwright
Visible demo: python scripts/automated_demo.py
Unattended check: python scripts/automated_demo.py --headless
Requires the trained API/UI stack to be running. Microsoft Edge is the default;
use --browser chromium after `python -m playwright install chromium` if needed.
"""

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from playwright.sync_api import Error as PlaywrightError
from playwright.sync_api import expect, sync_playwright

ROOT = Path(__file__).resolve().parents[1]
IMAGE = ROOT / "demo_images/01_true_negative_uid_1330.png"
INDICATION = "XXXX-year-old male with shortness of breath."
IMAGE_SHA256 = "f9d509d33b7ba2a322189f76022bf0a8b0675cc0d97637fd4fd763cf0f7feb61"
SECTIONS = [
    "Predicted Class: Normal",
    "Impression:",
    "Visual Findings:",
    "Clinical Context:",
    "Recommendation:",
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8501")
    parser.add_argument(
        "--headless", action="store_true", help="Run unattended and close on completion"
    )
    parser.add_argument(
        "--browser",
        choices=["msedge", "chromium"],
        default="msedge" if os.name == "nt" else "chromium",
    )
    parser.add_argument("--timeout", type=int, default=90, help="Per-step timeout in seconds")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "reports/automated_demo")
    parser.add_argument("--record-video", type=Path, help="Save a silent WebM fallback recording")
    args = parser.parse_args()
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    if not IMAGE.is_file():
        parser.error(f"Demo image not found: {IMAGE}")
    if hashlib.sha256(IMAGE.read_bytes()).hexdigest() != IMAGE_SHA256:
        parser.error("The demo image differs from the verified UID 1330 source image")
    folder = args.output_dir.resolve()
    folder.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as playwright:
        launch = {
            "headless": args.headless,
            "slow_mo": 250 if args.record_video or not args.headless else 0,
        }
        if args.browser == "msedge":
            launch["channel"] = "msedge"
        browser = playwright.chromium.launch(**launch)
        video_options = (
            {
                "record_video_dir": str(folder / "video_raw"),
                "record_video_size": {"width": 1280, "height": 940},
            }
            if args.record_video
            else {}
        )
        context = browser.new_context(
            viewport={"width": 1500, "height": 1100}, accept_downloads=True, **video_options
        )
        page = context.new_page()
        page.set_default_timeout(args.timeout * 1000)
        page_errors = []
        page.on("pageerror", lambda error: page_errors.append(str(error)))
        stage = "open UI and wait for model readiness"
        completed = False
        try:
            print(f"1/5 Opening {args.url} and waiting for Model ready...", flush=True)
            page.goto(args.url, wait_until="domcontentloaded")
            expect(page.get_by_text("Model ready", exact=True)).to_be_visible(
                timeout=args.timeout * 1000
            )
            if args.record_video:
                page.wait_for_timeout(2000)
            stage = "upload the X-ray"
            print(f"2/5 Uploading {IMAGE.name}...", flush=True)
            page.locator('input[type="file"]').set_input_files(str(IMAGE))
            # Wait for Streamlit's server rerun before entering the indication.
            expect(page.get_by_text("Uploaded image", exact=True)).to_be_visible(
                timeout=args.timeout * 1000
            )
            if args.record_video:
                page.wait_for_timeout(2000)
            stage = "enter indication and analyze"
            print(f"3/5 Entering indication: {INDICATION}", flush=True)
            field = page.get_by_label("Clinical indication", exact=True)
            field.fill(INDICATION)
            field.press("Tab")  # Commit the textarea value to Streamlit.
            if args.record_video:
                page.wait_for_timeout(2000)
            page.get_by_role("button", name="Analyze radiograph and text", exact=True).click()
            stage = "wait for the generated report"
            print("4/5 Waiting for the report...", flush=True)
            report = page.get_by_label("Rule-based report for clinician review", exact=True)
            expect(report).to_be_visible(timeout=args.timeout * 1000)
            report_text = report.input_value()
            for section in SECTIONS:
                if section not in report_text:
                    raise ValueError(f"Expected report section missing: {section}")
            page.screenshot(path=str(folder / "analysis.png"), full_page=True)
            report.scroll_into_view_if_needed()
            page.screenshot(path=str(folder / "report.png"), full_page=True)
            (folder / "report.txt").write_text(report_text, encoding="utf-8")
            if args.record_video:
                page.wait_for_timeout(5000)
            stage = "download and verify the result JSON"
            with page.expect_download() as pending:
                page.get_by_role("button", name="Download result JSON", exact=True).click()
            pending.value.save_as(str(folder / "result.json"))
            result = json.loads((folder / "result.json").read_text(encoding="utf-8"))
            if result["predicted_class"] != "Normal" or result["generated_report"] != report_text:
                raise ValueError("Downloaded result differs from the displayed Normal report")
            prediction = result.get("prediction", result)
            summary = {
                "status": "passed",
                "captured_at_utc": datetime.now(timezone.utc).isoformat(),
                "url": args.url,
                "uid": "1330",
                "image_sha256": IMAGE_SHA256,
                "indication": INDICATION,
                "predicted_class": result["predicted_class"],
                "probability": prediction["probability"],
                "model_version": prediction["model_version"],
                "report_sections_checked": SECTIONS,
                "page_errors": page_errors,
            }
            (folder / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
            completed = True
            print(f"5/5 PASS: Normal; score {prediction['probability']:.2%}.", flush=True)
            print(f"Report, JSON and screenshots saved to: {folder}", flush=True)
            if not args.headless:
                report.scroll_into_view_if_needed()
                input("Demo complete. Inspect the browser, then press Enter here to close it: ")
        except (PlaywrightError, AssertionError, ValueError, KeyError) as error:
            try:
                page.screenshot(path=str(folder / "failure.png"), full_page=True, timeout=5000)
            except PlaywrightError:
                pass
            (folder / "failure.json").write_text(
                json.dumps(
                    {
                        "status": "failed",
                        "stage": stage,
                        "error": str(error),
                        "page_errors": page_errors,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            raise
        finally:
            context.close()
            if completed and args.record_video:
                target = args.record_video.resolve()
                target.parent.mkdir(parents=True, exist_ok=True)
                page.video.save_as(str(target))
                (folder / "video_verification.json").write_text(
                    json.dumps(
                        {
                            **summary,
                            "video": str(args.record_video),
                            "bytes": target.stat().st_size,
                        },
                        indent=2,
                    ),
                    encoding="utf-8",
                )
            browser.close()


if __name__ == "__main__":
    try:
        main()
    except (PlaywrightError, AssertionError, ValueError, KeyError, OSError) as error:
        print(f"Demo failed: {error}", file=sys.stderr)
        print("Check Docker/API readiness and browser installation; see --help.", file=sys.stderr)
        sys.exit(1)
