from django.db import models

class ImageUpload(models.Model):
    STATUS_CHOICES = [
        ('PENDING', 'Pending'),
        ('COMPLETED', 'Completed'),
        ('REJECTED', 'Rejected'), # For "No Body"
        ('FAILED', 'Failed'),
    ]

    original_file = models.ImageField(upload_to='uploads/')
    processed_file = models.ImageField(upload_to='processed/', null=True, blank=True)
    mask_file = models.ImageField(upload_to='masks/', null=True, blank=True)
    
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='PENDING')
    error_log = models.TextField(blank=True)
    
    # Stores bbox, keypoints, and colors as JSON
    metadata = models.JSONField(null=True, blank=True) 
    
    created_at = models.DateTimeField(auto_now_add=True)