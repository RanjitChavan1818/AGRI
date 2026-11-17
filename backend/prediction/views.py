import logging

from rest_framework import status
from rest_framework.parsers import FormParser, MultiPartParser
from rest_framework.response import Response
from rest_framework.views import APIView

from .serializers import ImageUploadSerializer
from .services.model_service import ModelServiceError
from .services.prediction_service import predict_image
from .services.preprocessing import ImageValidationError

logger = logging.getLogger(__name__)


def error_response(code: str, message: str, status_code: int) -> Response:
    return Response({"success": False, "error": {"code": code, "message": message}}, status=status_code)


class PredictView(APIView):
    parser_classes = [MultiPartParser, FormParser]

    def post(self, request):
        if "image" not in request.FILES:
            return error_response("MISSING_IMAGE", "Please upload an image.", status.HTTP_400_BAD_REQUEST)
        serializer = ImageUploadSerializer(data=request.data)
        if not serializer.is_valid():
            return error_response("INVALID_IMAGE", "Please upload a valid JPG, PNG, or WEBP image.", status.HTTP_400_BAD_REQUEST)
        try:
            return Response(predict_image(serializer.validated_data["image"]), status=status.HTTP_200_OK)
        except ImageValidationError as exc:
            return error_response(exc.code, exc.message, exc.status_code)
        except ModelServiceError:
            logger.exception("Prediction model error")
            return error_response("MODEL_ERROR", "The prediction service is temporarily unavailable.", status.HTTP_500_INTERNAL_SERVER_ERROR)
        except Exception:
            logger.exception("Unexpected prediction endpoint error")
            return error_response("INTERNAL_ERROR", "An unexpected error occurred while processing the image.", status.HTTP_500_INTERNAL_SERVER_ERROR)
