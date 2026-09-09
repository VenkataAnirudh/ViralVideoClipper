import os
import urllib.request
import urllib.parse
import ssl
import logging
import re
import zipfile
import io

def get_font_dir() -> str:
    """Resolve absolute path to assets/fonts."""
    return os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(__file__)), "assets", "fonts"))

def download_font_from_google(font_filename: str, logger: logging.Logger) -> bool:
    """
    Attempt to download a missing font from Google Fonts.
    First tries downloading a ZIP archive from fonts.google.com and extracting the matching .ttf.
    If that returns HTML or fails, falls back to the CSS API to get the direct gstatic TTF URL.
    """
    fonts_dir = get_font_dir()
    os.makedirs(fonts_dir, exist_ok=True)
    dest_path = os.path.join(fonts_dir, font_filename)

    if os.path.exists(dest_path) and os.path.getsize(dest_path) > 10000:
        logger.info(f"Font already exists: {font_filename}")
        return True

    # 1. Parse family name and split PascalCase/CamelCase
    base_name = os.path.splitext(font_filename)[0]
    if "-" in base_name:
        family_part, weight_part = base_name.split("-", 1)
    else:
        family_part = base_name
        weight_part = "Regular"

    # Split camelcase/pascalcase: e.g. BarlowCondensed -> Barlow Condensed
    family_name = re.sub(r'(?<!^)(?=[A-Z])', ' ', family_part)
    
    logger.info(f"Guessed Google Font family name: '{family_name}' for '{font_filename}'")

    # SSL context bypass for Windows SSL errors
    ctx = ssl._create_unverified_context()

    # Strategy A: Download ZIP archive
    zip_url = f"https://fonts.google.com/download?family={urllib.parse.quote(family_name)}"
    logger.info(f"Attempting Strategy A (ZIP): Downloading from {zip_url}")
    try:
        req = urllib.request.Request(
            zip_url,
            headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
        )
        with urllib.request.urlopen(req, context=ctx, timeout=20) as response:
            content_type = response.headers.get('Content-Type', '')
            if "html" in content_type.lower():
                logger.warning("Strategy A returned HTML instead of a ZIP archive. Falling back to Strategy B.")
            else:
                zip_data = response.read()
                # Open zip in memory
                with zipfile.ZipFile(io.BytesIO(zip_data)) as z:
                    for info in z.infolist():
                        basename = os.path.basename(info.filename)
                        if basename.lower() == font_filename.lower():
                            # Extract matching file
                            with z.open(info) as src, open(dest_path, "wb") as dst:
                                dst.write(src.read())
                            logger.info(f"✓ Successfully extracted and saved font from ZIP: {dest_path}")
                            return True
                logger.warning(f"Could not find matching file '{font_filename}' in the downloaded ZIP archive.")
    except Exception as e:
        logger.warning(f"Strategy A (ZIP) failed: {e}")

    # Strategy B: Query official CSS API + gstatic CDN fallback
    logger.info("Attempting Strategy B (CSS API + gstatic fallback)...")
    
    # Map weight name to number
    weight_map = {
        "thin": "100",
        "extralight": "200",
        "ultralight": "200",
        "light": "300",
        "regular": "400",
        "medium": "500",
        "semibold": "600",
        "demibold": "600",
        "bold": "700",
        "extrabold": "800",
        "ultrabold": "800",
        "black": "900",
        "heavy": "900"
    }
    weight_number = weight_map.get(weight_part.lower(), "400")
    
    # Legacy User-Agent to get TTF URL from Google Fonts CSS API
    legacy_ua = "Mozilla/4.0 (compatible; MSIE 8.0; Windows NT 6.0)"
    css_url = f"https://fonts.googleapis.com/css?family={urllib.parse.quote(family_name)}:{weight_number}"
    
    try:
        req = urllib.request.Request(css_url, headers={'User-Agent': legacy_ua})
        with urllib.request.urlopen(req, context=ctx, timeout=15) as response:
            css_data = response.read().decode('utf-8')
            # Extract gstatic URL (handling legacy TTF URLs and direct font binary URLs)
            match = re.search(r'url\((["\']?)((https?:)?//fonts\.gstatic\.com/[^"\')]+)\1\)', css_data)
            if match:
                ttf_url = match.group(2)
                if ttf_url.startswith("//"):
                    ttf_url = "https:" + ttf_url
                logger.info(f"Found direct TTF URL: {ttf_url}")
                
                # Download TTF directly
                req_ttf = urllib.request.Request(ttf_url, headers={'User-Agent': 'Mozilla/5.0'})
                with urllib.request.urlopen(req_ttf, context=ctx, timeout=15) as ttf_response:
                    with open(dest_path, "wb") as f:
                        f.write(ttf_response.read())
                logger.info(f"✓ Successfully downloaded font from gstatic: {dest_path}")
                return True
            else:
                logger.warning(f"Could not parse TTF URL from CSS. CSS response: {css_data}")
    except Exception as e:
        logger.error(f"Strategy B failed: {e}")

    logger.error(f"Failed to auto-download font '{font_filename}' from Google Fonts.")
    return False
