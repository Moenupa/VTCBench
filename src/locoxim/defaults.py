import os

# PIL stuff, ref:
# https://pillow.readthedocs.io/en/stable/handbook/image-file-formats.html#fully-supported-formats
PIL_SAVE_FORMAT = os.environ.get("PIL_SAVE_FORMAT", "webp").lower()

# Default JPEG quality for image saving
JPEG_QUALITY = int(os.environ.get("JPEG_QUALITY", 80))
WEBP_QUALITY = int(os.environ.get("WEBP_QUALITY", 80))

PIL_SAVE_KWARGS = {
    "jpeg": {"quality": JPEG_QUALITY},
    "png": {},
    "webp": {"quality": WEBP_QUALITY},
}
