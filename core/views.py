from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework import status
from .serializers import ImageUploadSerializer
from .tasks import process_image_pipeline

class ImageUploadView(APIView):
    # Tells DRF to expect file uploads (multipart/form-data)
    parser_classes = (MultiPartParser, FormParser)

    def post(self, request, *args, **kwargs):
        serializer = ImageUploadSerializer(data=request.data)
        
        if serializer.is_valid():
            # 1. Save to DB immediately
            entry = serializer.save()
            
            # 2. Trigger Celery Task
            process_image_pipeline.delay(entry.id)
            
            return Response({
                "id": entry.id, 
                "status": "PENDING", 
                "message": "Upload successful, processing started."
            }, status=status.HTTP_201_CREATED)
            
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)