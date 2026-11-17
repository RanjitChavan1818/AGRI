"""Singleton access to the locally stored Keras classification model."""
import logging
from pathlib import Path
from threading import Lock

import numpy as np
from django.conf import settings

from .class_names import validate_class_count

logger = logging.getLogger(__name__)
_model = None
_model_lock = Lock()


class ModelServiceError(RuntimeError):
    """An error while loading or running the local ML model."""


def _model_path() -> Path:
    return Path(settings.BASE_DIR) / "models" / "corn_disease_efficientnetb0.keras"


def _load_model_from_disk():
    model_path = _model_path()
    if not model_path.is_file():
        raise ModelServiceError(f"Model file was not found at {model_path}.")
    try:
        import tensorflow as tf

        model = tf.keras.models.load_model(model_path, compile=False)
        validate_class_count(int(model.output_shape[-1]))
    except ModelServiceError:
        raise
    except Exception as exc:
        raise ModelServiceError("The prediction model could not be loaded.") from exc
    logger.info("Corn disease model loaded successfully from %s", model_path)
    return model


def get_model():
    """Return the process-local model, loading it once when needed."""
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:
                try:
                    _model = _load_model_from_disk()
                except ModelServiceError:
                    logger.exception("Failed to load the corn disease model")
                    raise
    return _model


def predict(input_array: np.ndarray) -> np.ndarray:
    try:
        return np.asarray(get_model().predict(input_array, verbose=0))
    except ModelServiceError:
        raise
    except Exception as exc:
        logger.exception("Corn disease model prediction failed")
        raise ModelServiceError("The prediction model could not process this image.") from exc
