"""Business logic that turns an image upload into a prediction response."""
import logging

import numpy as np

from . import model_service
from .class_names import CLASS_NAMES, DISPLAY_NAMES, validate_class_count
from .gradcam import generate_data_url
from .preprocessing import preprocess_image

logger = logging.getLogger(__name__)


def predict_image(uploaded_file) -> dict:
    input_array = preprocess_image(uploaded_file)
    probabilities = np.asarray(model_service.predict(input_array), dtype=np.float64).reshape(-1)
    try:
        validate_class_count(probabilities.size)
    except ValueError as exc:
        raise model_service.ModelServiceError("The model returned an unexpected output.") from exc
    if not np.all(np.isfinite(probabilities)) or not np.isclose(probabilities.sum(), 1.0, atol=0.01):
        raise model_service.ModelServiceError("The model returned invalid probabilities.")

    predicted_index = int(np.argmax(probabilities))
    probability_items = [
        {"class": class_name, "label": DISPLAY_NAMES[class_name], "confidence": round(float(probability * 100), 2)}
        for class_name, probability in zip(CLASS_NAMES, probabilities, strict=True)
    ]
    predicted_class = CLASS_NAMES[predicted_index]
    payload = {
        "success": True,
        "prediction": {
            "class": predicted_class,
            "label": DISPLAY_NAMES[predicted_class],
            "confidence": probability_items[predicted_index]["confidence"],
        },
        "probabilities": probability_items,
    }
    try:
        payload["gradcam"] = {
            "available": True,
            "image": generate_data_url(model_service.get_model(), input_array, predicted_index, uploaded_file),
        }
    except Exception:
        logger.exception("Grad-CAM generation failed; returning prediction without visualization")
        payload["gradcam"] = {"available": False, "image": None}
    return payload
