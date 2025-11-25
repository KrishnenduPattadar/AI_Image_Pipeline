from django.contrib import admin
from django.utils.html import format_html
from .models import ImageUpload

@admin.register(ImageUpload)
class ImageUploadAdmin(admin.ModelAdmin):
    # Fields to display on the main list page
    list_display = [
        'id', 
        'status', 
        'original_image_tag', 
        'processed_image_tag', 
        'get_confidence', 
        'created_at'
    ]
    
    # Fields to filter the list by
    list_filter = ['status', 'created_at']
    
    # Fields to search across
    search_fields = ['id', 'error_log']
    
    # Fields to display on the individual detail page
    fields = [
        'original_file', 
        'status', 
        'error_log', 
        'original_image_tag',
        'processed_file',
        'mask_file',
        'metadata'
    ]
    
    # Make metadata readonly, since it's an output field
    readonly_fields = [
        'original_image_tag', 
        'processed_file', 
        'mask_file', 
        'metadata'
    ]

    # --- Custom Methods for Display ---

    def original_image_tag(self, obj):
        """Displays a thumbnail of the original uploaded image."""
        if obj.original_file:
            return format_html(
                '<img src="{}" width="80" height="80" style="object-fit: cover; border-radius: 4px;" />', 
                obj.original_file.url
            )
        return "No Image"
    original_image_tag.short_description = 'Original'

    def processed_image_tag(self, obj):
        """Displays a thumbnail of the transparent PNG result."""
        if obj.processed_file:
            return format_html(
                '<img src="{}" width="80" height="80" style="object-fit: cover; border-radius: 4px; background-color: #f0f0f0;" />', 
                obj.processed_file.url
            )
        return "N/A"
    processed_image_tag.short_description = 'Processed PNG'

    def get_confidence(self, obj):
        """Extracts and displays the detection confidence from the metadata."""
        if obj.metadata and 'detection_confidence' in obj.metadata:
            # Format as percentage (e.g., 99.99%)
            confidence = obj.metadata['detection_confidence'] * 100
            return f"{confidence:.2f}%"
        return "---"
    get_confidence.short_description = 'Confidence'