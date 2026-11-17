import io
import base64
from unittest.mock import patch

import numpy as np
from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import TestCase
from PIL import Image
from rest_framework.test import APIClient

from prediction.services import model_service
from prediction.services.class_names import CLASS_NAMES, DISPLAY_NAMES
from prediction.services.gradcam import generate_heatmap, overlay_as_data_url
from prediction.services.model_service import ModelServiceError
from prediction.services.prediction_service import predict_image
from prediction.services.preprocessing import ImageValidationError, preprocess_image

PREDICT_URL = "/api/predict/"
MODEL_OUTPUT = np.array([[0.0006, 0.9992, 0.0002, 0.0]], dtype=np.float32)


def image_upload(image_format: str, name: str, color=(200, 100, 50)) -> SimpleUploadedFile:
    buffer = io.BytesIO()
    Image.new("RGB", (48, 32), color=color).save(buffer, format=image_format)
    content_type = {"JPEG": "image/jpeg", "PNG": "image/png", "WEBP": "image/webp", "GIF": "image/gif"}[image_format]
    return SimpleUploadedFile(name, buffer.getvalue(), content_type=content_type)


class PredictionEndpointTests(TestCase):
    def setUp(self):
        self.client = APIClient()
        self.model_patch = patch("prediction.services.prediction_service.model_service.predict", return_value=MODEL_OUTPUT)
        self.model_patch.start()
        self.addCleanup(self.model_patch.stop)
        self.get_model_patch = patch("prediction.services.prediction_service.model_service.get_model", return_value=object())
        self.get_model_patch.start()
        self.addCleanup(self.get_model_patch.stop)
        self.gradcam_patch = patch(
            "prediction.services.prediction_service.generate_data_url",
            return_value="data:image/jpeg;base64,dGVzdA==",
        )
        self.gradcam_patch.start()
        self.addCleanup(self.gradcam_patch.stop)

    def post_image(self, image):
        return self.client.post(PREDICT_URL, {"image": image}, format="multipart")

    def test_endpoint_exists(self):
        self.assertEqual(self.client.get(PREDICT_URL).status_code, 405)

    def test_post_without_image_is_rejected(self):
        response = self.client.post(PREDICT_URL, {}, format="multipart")
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["error"]["code"], "MISSING_IMAGE")

    def test_empty_file_is_rejected(self):
        response = self.post_image(SimpleUploadedFile("empty.jpg", b"", content_type="image/jpeg"))
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["error"]["code"], "INVALID_IMAGE")

    def test_invalid_file_is_rejected(self):
        response = self.post_image(SimpleUploadedFile("leaf.jpg", b"not an image", content_type="image/jpeg"))
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["error"]["code"], "INVALID_IMAGE")

    def test_corrupted_image_is_rejected(self):
        response = self.post_image(SimpleUploadedFile("leaf.png", b"\x89PNG\r\n\x1a\ncorrupt", content_type="image/png"))
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["error"]["code"], "INVALID_IMAGE")

    def test_unsupported_format_is_rejected(self):
        response = self.post_image(image_upload("GIF", "leaf.gif"))
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["error"]["code"], "UNSUPPORTED_FORMAT")

    def test_oversized_file_is_rejected(self):
        upload = SimpleUploadedFile("large.jpg", b"x" * (5 * 1024 * 1024 + 1), content_type="image/jpeg")
        response = self.post_image(upload)
        self.assertEqual(response.status_code, 413)
        self.assertEqual(response.json()["error"]["code"], "FILE_TOO_LARGE")

    def test_valid_jpeg_png_and_webp_images(self):
        for image_format, name in (("JPEG", "leaf.jpg"), ("PNG", "leaf.png"), ("WEBP", "leaf.webp")):
            with self.subTest(image_format=image_format):
                response = self.post_image(image_upload(image_format, name))
                self.assertEqual(response.status_code, 200)

    def test_prediction_response_structure_and_ranges(self):
        response = self.post_image(image_upload("PNG", "leaf.png"))
        payload = response.json()
        self.assertTrue(payload["success"])
        self.assertEqual(set(payload), {"success", "prediction", "probabilities", "gradcam"})
        self.assertEqual(len(payload["probabilities"]), 4)
        self.assertIn(payload["prediction"]["class"], CLASS_NAMES)
        self.assertEqual(payload["prediction"]["label"], DISPLAY_NAMES[payload["prediction"]["class"]])
        self.assertGreaterEqual(payload["prediction"]["confidence"], 0)
        self.assertLessEqual(payload["prediction"]["confidence"], 100)
        self.assertAlmostEqual(sum(item["confidence"] for item in payload["probabilities"]), 100, places=2)
        for item in payload["probabilities"]:
            self.assertIn(item["class"], CLASS_NAMES)
            self.assertEqual(item["label"], DISPLAY_NAMES[item["class"]])
            self.assertGreaterEqual(item["confidence"], 0)
            self.assertLessEqual(item["confidence"], 100)
        self.assertTrue(payload["gradcam"]["available"])
        self.assertTrue(payload["gradcam"]["image"].startswith("data:image/jpeg;base64,"))

    def test_fixed_class_index_mapping_is_preserved(self):
        response = self.post_image(image_upload("JPEG", "leaf.jpg"))
        payload = response.json()
        self.assertEqual(payload["prediction"]["class"], CLASS_NAMES[1])
        self.assertEqual(payload["prediction"]["label"], "Common Rust")
        self.assertEqual([item["class"] for item in payload["probabilities"]], CLASS_NAMES)

    @patch("prediction.views.predict_image", side_effect=ModelServiceError("failed"))
    def test_model_service_errors_are_safe(self, _mock_predict):
        response = self.post_image(image_upload("JPEG", "leaf.jpg"))
        self.assertEqual(response.status_code, 500)
        self.assertEqual(response.json()["error"]["code"], "MODEL_ERROR")

    def test_gradcam_failure_preserves_prediction_response(self):
        self.gradcam_patch.stop()
        with patch("prediction.services.prediction_service.generate_data_url", side_effect=RuntimeError("Grad-CAM failed")):
            response = self.post_image(image_upload("JPEG", "leaf.jpg"))
        payload = response.json()
        self.assertEqual(response.status_code, 200)
        self.assertTrue(payload["success"])
        self.assertIn("prediction", payload)
        self.assertEqual(len(payload["probabilities"]), 4)
        self.assertEqual(payload["gradcam"], {"available": False, "image": None})


class PreprocessingAndServiceTests(TestCase):
    def test_preprocessing_returns_rgb_float32_batch_without_external_rescaling(self):
        batch = preprocess_image(image_upload("PNG", "leaf.png", color=(200, 100, 50)))
        self.assertEqual(batch.shape, (1, 224, 224, 3))
        self.assertEqual(batch.dtype, np.float32)
        self.assertAlmostEqual(float(batch[0, 0, 0, 0]), 200.0)
        self.assertGreater(float(batch.max()), 1.0)

    @patch("prediction.services.prediction_service.model_service.predict", return_value=np.array([[0.1, 0.2, 0.6, 0.1]]))
    def test_prediction_service_uses_argmax_and_exact_mapping(self, _mock_predict):
        payload = predict_image(image_upload("JPEG", "leaf.jpg"))
        self.assertEqual(payload["prediction"]["class"], CLASS_NAMES[2])
        self.assertEqual(payload["prediction"]["label"], "Northern Leaf Blight")

    def test_model_service_load_error_is_reported(self):
        original_model = model_service._model
        model_service._model = None
        self.addCleanup(setattr, model_service, "_model", original_model)
        with patch("prediction.services.model_service._load_model_from_disk", side_effect=ModelServiceError("missing")):
            with self.assertRaises(ModelServiceError):
                model_service.get_model()

    def test_model_is_cached_after_the_first_load(self):
        original_model = model_service._model
        cached_model = object()
        model_service._model = None
        self.addCleanup(setattr, model_service, "_model", original_model)
        with patch("prediction.services.model_service._load_model_from_disk", return_value=cached_model) as loader:
            self.assertIs(model_service.get_model(), cached_model)
            self.assertIs(model_service.get_model(), cached_model)
            loader.assert_called_once()

    def test_invalid_image_raises_validation_error(self):
        with self.assertRaises(ImageValidationError):
            preprocess_image(SimpleUploadedFile("leaf.jpg", b"not an image", content_type="image/jpeg"))


class GradCamUtilityTests(TestCase):
    @staticmethod
    def _test_model():
        import tensorflow as tf

        inputs = tf.keras.Input(shape=(224, 224, 3))
        augmentation = tf.keras.Sequential([tf.keras.layers.Activation("linear")], name="data_augmentation")
        nested_inputs = tf.keras.Input(shape=(224, 224, 3))
        nested_output = tf.keras.layers.Conv2D(
            1280,
            3,
            strides=32,
            kernel_initializer="ones",
            name="conv",
        )(nested_inputs)
        nested_output = tf.keras.layers.Activation("relu", name="top_activation")(nested_output)
        efficientnet = tf.keras.Model(nested_inputs, nested_output, name="efficientnetb0")
        value = augmentation(inputs)
        value = efficientnet(value)
        value = tf.keras.layers.GlobalAveragePooling2D(name="global_average_pooling2d")(value)
        value = tf.keras.layers.Dropout(0.0, name="dropout")(value)
        outputs = tf.keras.layers.Dense(4, kernel_initializer="ones", name="dense")(value)
        return tf.keras.Model(inputs, outputs)

    def test_heatmap_is_finite_normalized_and_expected_shape(self):
        heatmap = generate_heatmap(self._test_model(), np.ones((1, 224, 224, 3), dtype=np.float32), 0)
        self.assertEqual(heatmap.shape, (7, 7))
        self.assertTrue(np.all(np.isfinite(heatmap)))
        self.assertGreaterEqual(float(heatmap.min()), 0.0)
        self.assertLessEqual(float(heatmap.max()), 1.0)
        self.assertGreater(float(heatmap.max()), 0.0)

    def test_overlay_is_a_warm_heatmap_jpeg_at_the_original_dimensions(self):
        upload = image_upload("PNG", "leaf.png", color=(20, 100, 30))
        heatmap = np.zeros((7, 7), dtype=np.float32)
        heatmap[2:5, 2:5] = 1.0
        data_url = overlay_as_data_url(upload, heatmap)
        encoded = data_url.removeprefix("data:image/jpeg;base64,")
        with Image.open(io.BytesIO(base64.b64decode(encoded))) as overlay:
            self.assertEqual(overlay.format, "JPEG")
            self.assertEqual(overlay.size, (48, 32))
            pixels = np.asarray(overlay.convert("RGB"))
            warm_pixels = (pixels[:, :, 0] > pixels[:, :, 1]) & (pixels[:, :, 0] > pixels[:, :, 2])
            self.assertTrue(warm_pixels.any())
