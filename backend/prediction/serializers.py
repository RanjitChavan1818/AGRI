from rest_framework import serializers


class ImageUploadSerializer(serializers.Serializer):
    """Validate multipart request shape; image-content validation is in services."""

    image = serializers.FileField(required=False, allow_empty_file=True)
