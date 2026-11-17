"""Safe, in-memory image validation and preprocessing."""
import io

import numpy as np
from PIL import Image, UnidentifiedImageError

IMAGE_SIZE = (224, 224)
MAX_IMAGE_SIZE_BYTES = 5 * 1024 * 1024
SUPPORTED_FORMATS = {"JPEG", "PNG", "WEBP"}


class ImageValidationError(ValueError):
    def __init__(self, code: str, message: str, status_code: int = 400) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code


def _rewind(uploaded_file) -> None:
    if hasattr(uploaded_file, "seek"):
        uploaded_file.seek(0)


def preprocess_image(uploaded_file) -> np.ndarray:
    """Validate an upload and create a float32 RGB batch without rescaling."""
    if uploaded_file is None:
        raise ImageValidationError("MISSING_IMAGE", "Please upload an image.")
    size = getattr(uploaded_file, "size", None)
    if size is None or size == 0:
        raise ImageValidationError("INVALID_IMAGE", "The uploaded image is empty.")
    if size > MAX_IMAGE_SIZE_BYTES:
        raise ImageValidationError("FILE_TOO_LARGE", "Image files must be 5 MB or smaller.", 413)

    try:
        _rewind(uploaded_file)
        image_bytes = uploaded_file.read()
        if not image_bytes:
            raise ImageValidationError("INVALID_IMAGE", "The uploaded image is empty.")
        with Image.open(io.BytesIO(image_bytes)) as verification_image:
            image_format = verification_image.format
            verification_image.verify()
        if image_format not in SUPPORTED_FORMATS:
            raise ImageValidationError("UNSUPPORTED_FORMAT", "Please upload a JPG, PNG, or WEBP image.")
        with Image.open(io.BytesIO(image_bytes)) as source_image:
            source_image.load()
            resized_image = source_image.convert("RGB").resize(IMAGE_SIZE, Image.Resampling.LANCZOS)
            array = np.asarray(resized_image, dtype=np.float32)
    except ImageValidationError:
        raise
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise ImageValidationError("INVALID_IMAGE", "Please upload a valid JPG, PNG, or WEBP image.") from exc
    finally:
        _rewind(uploaded_file)
    return np.expand_dims(array, axis=0)
