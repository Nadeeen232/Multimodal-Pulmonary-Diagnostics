"""Capture the actual running Docker web interfaces with installed Microsoft Edge."""

import argparse
import csv
import json
import os
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--browser",
        choices=["msedge", "chrome", "chromium"],
        default="msedge" if os.name == "nt" else "chromium",
    )
    args = parser.parse_args()
    output = ROOT / "reports"
    with (ROOT / "demo_images/demo_cases.csv").open(encoding="utf-8", newline="") as handle:
        case = next(csv.DictReader(handle))
    with sync_playwright() as playwright:
        options = {"headless": True}
        if args.browser != "chromium":
            options["channel"] = args.browser
        browser = playwright.chromium.launch(**options)
        page = browser.new_page(viewport={"width": 1500, "height": 1500}, device_scale_factor=1)
        errors = []
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.goto("http://127.0.0.1:8080/docs", wait_until="networkidle")
        page.locator(".swagger-ui .opblock").first.wait_for(timeout=60000)
        page.screenshot(path=str(output / "swagger_screenshot.png"), full_page=True)
        page.goto("http://127.0.0.1:8501", wait_until="domcontentloaded")
        page.get_by_text("Model ready", exact=True).wait_for(timeout=60000)
        page.locator('input[type="file"]').set_input_files(
            str(ROOT / "demo_images" / case["filename"])
        )
        page.get_by_text("Uploaded image", exact=True).wait_for(timeout=60000)
        page.get_by_label("Clinical indication", exact=True).fill(case["indication"])
        page.get_by_label("Clinical indication", exact=True).press("Tab")
        page.get_by_label("Comparison context (optional)", exact=True).fill(
            case.get("comparison", "")
        )
        page.get_by_label("Comparison context (optional)", exact=True).press("Tab")
        page.get_by_role("button", name="Analyze radiograph and text", exact=True).click()
        report = page.get_by_label("Rule-based report for clinician review", exact=True)
        report.wait_for(timeout=60000)
        assert "Predicted Class: Normal" in report.input_value()
        page.screenshot(path=str(output / "ui_screenshot.png"), full_page=True)
        report.scroll_into_view_if_needed()
        page.screenshot(path=str(output / "report_screenshot.png"), full_page=True)
        browser.close()
    (output / "evidence/browser_capture.json").write_text(
        json.dumps(
            {
                "status": "passed",
                "browser": f"{args.browser} (Playwright, isolated headless context)",
                "uid": case["uid"],
                "routes": ["http://127.0.0.1:8080/docs", "http://127.0.0.1:8501"],
                "checks": [
                    "Swagger operations visible",
                    "Model ready",
                    "real upload and analysis",
                    "generated report visible",
                ],
                "page_errors": errors,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print("Saved real Swagger and Streamlit screenshots, including live report analysis.")


if __name__ == "__main__":
    main()
