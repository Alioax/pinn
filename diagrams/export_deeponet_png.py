"""
Export diagrams/deeponet_architecture.html to PNG at maximum quality.
Uses Playwright to render the page at 2x device scale and save a full-page screenshot.

Run once: pip install playwright && playwright install chromium
Then: python export_deeponet_png.py
"""

import os
from pathlib import Path

def main():
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print("Install Playwright: pip install playwright && playwright install chromium")
        raise SystemExit(1)

    base = Path(__file__).resolve().parent
    html_path = base / "deeponet_architecture.html"
    out_path = base / "deeponet_architecture.png"

    if not html_path.exists():
        print(f"Not found: {html_path}")
        raise SystemExit(1)

    # High-DPI scale for maximum quality (3x: 3600x1740 PNG)
    viewport_width = 1200
    viewport_height = 580   # room for title + diagram
    device_scale_factor = 3

    with sync_playwright() as p:
        browser = p.chromium.launch()
        context = browser.new_context(
            viewport={"width": viewport_width, "height": viewport_height},
            device_scale_factor=device_scale_factor,
        )
        page = context.new_page()
        page.goto(html_path.as_uri())
        page.screenshot(path=str(out_path), full_page=True)
        browser.close()

    print(f"Saved: {out_path}")
    print(f"Resolution: {viewport_width * device_scale_factor}x{viewport_height * device_scale_factor} ({device_scale_factor}x scale)")


if __name__ == "__main__":
    main()
