"""
One-time font downloader for Video Clipper caption styles.

Called by install.bat. Safe to re-run; existing font files are skipped.
"""

import os
import sys
import logging
from pipeline.font_downloader import download_font_from_google

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("font_downloader")

FONTS = [
    "Anton-Regular.ttf",
    "BarlowCondensed-Black.ttf",
    "Oswald-Bold.ttf",
    "Urbanist-ExtraBold.ttf",
    "Lexend-ExtraBold.ttf"
]

print("Starting Google Font assets downloader...")
for font_filename in FONTS:
    try:
        success = download_font_from_google(font_filename, logger)
        if success:
            print(f"  [OK] Checked/Downloaded: {font_filename}")
        else:
            print(f"  [!] Failed to download: {font_filename}")
    except Exception as e:
        print(f"  [ERROR] {font_filename}: {e}")

print("")
print("Font check complete.")
