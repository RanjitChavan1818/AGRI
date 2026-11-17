"""In-memory Grad-CAM generation for the saved corn classifier."""
import base64
import io

import numpy as np
import tensorflow as tf
from PIL import Image


_NESTED_MODEL_NAME = "efficientnetb0"
_TARGET_LAYER_NAME = "top_activation"


def generate_heatmap(model, input_array: np.ndarray, class_index: int) -> np.ndarray:
    """Return a normalized 7x7 Grad-CAM heatmap for ``class_index``.

    The outer model is deliberately not re-wrapped as a Functional model.  Keras
    cannot reliably join a nested symbolic layer output to the outer graph after
    deserialization.  Calling the existing outer layers in order keeps the
    activation and the selected class score in the same GradientTape graph.
    """
    nested_model = model.get_layer(_NESTED_MODEL_NAME)
    target_layer = nested_model.get_layer(_TARGET_LAYER_NAME)
    if nested_model.output is not target_layer.output:
        raise ValueError("The nested EfficientNetB0 output is not top_activation.")

    nested_index = model.layers.index(nested_model)
    if nested_index == 0:
        raise ValueError("The outer model has no input processing layers.")

    with tf.GradientTape() as tape:
        value = input_array
        for layer in model.layers[1:nested_index]:
            value = layer(value, training=False)
        activations = nested_model(value, training=False)
        value = activations
        for layer in model.layers[nested_index + 1 :]:
            value = layer(value, training=False)
        target_score = value[:, class_index]

    gradients = tape.gradient(target_score, activations)
    if gradients is None:
        raise ValueError("No gradients were produced for efficientnetb0/top_activation.")
    if tuple(activations.shape[1:]) != (7, 7, 1280):
        raise ValueError(f"Unexpected top_activation shape: {tuple(activations.shape)}")
    if tuple(gradients.shape) != tuple(activations.shape):
        raise ValueError(f"Gradient shape does not match activations: {tuple(gradients.shape)}")

    channel_weights = tf.reduce_mean(gradients, axis=(0, 1, 2))
    heatmap = tf.reduce_sum(activations[0] * channel_weights, axis=-1)
    heatmap = tf.nn.relu(heatmap)
    maximum = tf.reduce_max(heatmap)
    heatmap = tf.where(maximum > 0, heatmap / maximum, heatmap)
    result = np.asarray(heatmap.numpy(), dtype=np.float32)
    if result.shape != (7, 7) or not np.all(np.isfinite(result)):
        raise ValueError("Grad-CAM produced an invalid heatmap.")
    return result


def overlay_as_data_url(uploaded_file, heatmap: np.ndarray) -> str:
    """Blend a warm, transparent Grad-CAM heatmap over the original upload."""
    if heatmap.shape != (7, 7):
        raise ValueError(f"Expected a 7x7 heatmap, received {heatmap.shape}.")
    uploaded_file.seek(0)
    try:
        image_bytes = uploaded_file.read()
        with Image.open(io.BytesIO(image_bytes)) as source:
            original = source.convert("RGB")
            heatmap_image = Image.fromarray(np.uint8(np.clip(heatmap, 0, 1) * 255), mode="L")
            heatmap_image = heatmap_image.resize(original.size, Image.Resampling.BICUBIC)
            activation = np.asarray(heatmap_image, dtype=np.float32) / 255.0
            colors = np.empty((*activation.shape, 3), dtype=np.float32)
            colors[:, :, 0] = 255.0 - 65.0 * activation
            colors[:, :, 1] = 220.0 * (1.0 - activation)
            colors[:, :, 2] = 15.0 * (1.0 - activation)
            alpha = np.clip((activation - 0.04) / 0.96, 0.0, 1.0) * 0.50
            pixels = np.asarray(original, dtype=np.float32)
            overlay = Image.fromarray(np.uint8(pixels * (1.0 - alpha[:, :, None]) + colors * alpha[:, :, None]), mode="RGB")
            output = io.BytesIO()
            overlay.save(output, format="JPEG", quality=95)
    finally:
        uploaded_file.seek(0)
    encoded = base64.b64encode(output.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def generate_data_url(model, input_array: np.ndarray, class_index: int, uploaded_file) -> str:
    """Build the Grad-CAM heatmap overlay without creating files."""
    return overlay_as_data_url(uploaded_file, generate_heatmap(model, input_array, class_index))
