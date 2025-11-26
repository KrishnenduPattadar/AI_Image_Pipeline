# from celery import shared_task
# from .models import ImageUpload
# from django.conf import settings
# from .ai_engine import (
#     validate_image, analyze_human_pose, 
#     remove_background_hq, get_dominant_colors, generate_clothing_mask, get_image_dimensions, estimate_age_sex
# )
# import os
# import cv2
# from kombu.exceptions import OperationalError # Required for smart retries
# banner = r"""
# _
#        / \      _-'
#      _/|  \-''- _ /
# __-' { |          \
#     /             \        '''
#     /       "o.  |o }       Server Side AI 
#     |            \ ;            Processing Task 
#                   ',                        '''
#        \_         __\
#          ''-_    \.//
#            / '-____'
#           /
#         _'
#       _-'
# ░▒▓███▀▒ ▒▒   ▓▒█░░ ▒░   ▒ ▒ ░ ▒░   ▒ ▒ ░░ ▒░ ░░ ▒▓ ░▒▓░
# ▒░▒   ░   ▒   ▒▒ ░░ ░░   ░ ▒░░ ░░   ░ ▒░ ░ ░  ░  ░▒ ░ ▒░
#  ░    ░   ░   ▒      ░   ░ ░    ░   ░ ░    ░     ░░   ░ 
#  ░            ░  ░         ░          ░    ░  ░   ░     
                                            
#          Running KRISH cmd
# """

# print(banner)

# @shared_task(bind=True, max_retries=2, time_limit=120) # Timeout 2 mins
# def process_image_pipeline(self, upload_id):
#     entry = ImageUpload.objects.get(id=upload_id)
#     original_path = entry.original_file.path
    

#     # 🚨 CRUCIAL: Get image dimensions (w, h) before pixel conversion
#     w, h = get_image_dimensions(original_path)

#     # 1. Validate
#     is_valid, msg = validate_image(entry.original_file.path)
#     if not is_valid:
#         entry.status = 'FAILED'
#         entry.error_log = msg
#         entry.save()
#         return

#     # 2. Detect Human (Reject if None)
#     human_data = analyze_human_pose(entry.original_file.path)
#     if not human_data:
#         entry.status = 'REJECTED'
#         entry.error_log = "No human body detected"
#         entry.save()
#         return

#     try:
#         # --- Directory Creation (Recommended Robustness) ---
#         import os
#         os.makedirs(os.path.join(settings.MEDIA_ROOT, 'processed'), exist_ok=True)
#         os.makedirs(os.path.join(settings.MEDIA_ROOT, 'masks'), exist_ok=True)
        
#         # 3. Remove Background
#         processed_filename = f"transparent_{entry.id}.png"
#         processed_path = os.path.join(settings.MEDIA_ROOT, 'processed', processed_filename)
#         remove_background_hq(entry.original_file.path, processed_path)
        
#         # 4. Clothing Mask
#         mask_filename = f"mask_{entry.id}.png"
#         mask_path = os.path.join(settings.MEDIA_ROOT, 'masks', mask_filename) # Correct mask path construction
#         generate_clothing_mask(entry.original_file.path, mask_path)

#         # 5. Colors (FIXED CALL: Pass the mask_path)
#         colors = get_dominant_colors(entry.original_file.path, mask_path) # <-- CORRECTED CALL
# # --- AGE/SEX ESTIMATION (OPTIONAL GOAL) ---
#         # Calculate accurate Face BBox using keypoint data and image dimensions (w, h)
#         kp_nose = human_data['keypoints'][0]
#         kp_right_eye = human_data['keypoints'][3]
#         kp_left_eye = human_data['keypoints'][6]

#         # Calculate face dimensions based on eye distance
#         eye_distance_norm = abs(kp_right_eye['x'] - kp_left_eye['x'])
#         face_width_norm = eye_distance_norm * 2.5
#         face_top_y_norm = kp_nose['y'] - (face_width_norm * 0.45) # Estimate top of head

#         # Convert to Pixel BBox (x, y, w, h)
#         x_center_px = int(kp_nose['x'] * w)
#         y_start_px = int(face_top_y_norm * h) 
#         width_px = int(face_width_norm * w)
#         height_px = int(face_width_norm * 1.2) # Use a fixed height ratio

#         x_start_px = x_center_px - (width_px // 2)
#         x_start_px = max(0, x_start_px) # Clamp to 0
#         y_start_px = max(0, y_start_px)
        
#         face_bbox_pixels = (x_start_px, y_start_px, width_px, height_px)
        
#         estimated_age, estimated_gender = estimate_age_sex(original_path, face_bbox_pixels)

#         # --- FINAL SAVE AND METADATA ASSEMBLY ---
#         entry.processed_file = os.path.join('processed', processed_filename)
#         entry.mask_file = os.path.join('masks', mask_filename) 
        
#         entry.metadata = {
#             "bbox": human_data['bbox'],
#             "keypoints": human_data['keypoints'],
#             "detection_confidence": human_data['confidence'],
#             "dominant_colors": colors,
#             "estimated_age": estimated_age,
#             "estimated_gender": estimated_gender
#         }
#         entry.status = 'COMPLETED'
#         entry.save()

#     # --- 🛡️ SMART ERROR HANDLING (Required Refinement) ---
#     # CATCH 1: TRANSIENT ERRORS (Retry)
#     except (IOError, OSError, OperationalError) as e:
#         # Retry only if it's a temporary issue (like Redis/file locking)
#         if self.request.retries < self.max_retries:
#             entry.error_log = f"Transient error (Retrying {self.request.retries + 1}): {e}"
#             entry.save()
#             raise self.retry(exc=e, countdown=5)
#         else:
#             # If maximum retries reached, mark as failed
#             entry.status = 'FAILED'; entry.error_log = f"Max retries reached: {e}"; entry.save();
            
#     # CATCH 2: PERMANENT/STRUCTURAL ERRORS (Fail Immediately)
#     except Exception as e:
#         # Catches Model loading failures, structural logic errors, etc.
#         entry.status = 'FAILED'
#         entry.error_log = f"Permanent processing error: {e}"
#         entry.save()


# new modification 

from celery import shared_task
from .models import ImageUpload
from django.conf import settings
from .ai_engine import (
    validate_image, analyze_human_pose, 
    remove_background_hq, get_dominant_colors, generate_clothing_mask, 
    get_image_dimensions, estimate_age_sex
)
import os
from kombu.exceptions import OperationalError

@shared_task(bind=True, max_retries=2, time_limit=120) 
def process_image_pipeline(self, upload_id):
    entry = ImageUpload.objects.get(id=upload_id)
    original_path = entry.original_file.path
    
    # 0. Get Dimensions
    w, h = get_image_dimensions(original_path)

    # 1. Validate
    is_valid, msg = validate_image(original_path)
    if not is_valid:
        entry.status = 'FAILED'; entry.error_log = msg; entry.save(); return

    # 2. Detect Human
    human_data = analyze_human_pose(original_path)
    if not human_data:
        entry.status = 'REJECTED'; entry.error_log = "No human body detected"; entry.save(); return

    try:
        # Setup Directories
        os.makedirs(os.path.join(settings.MEDIA_ROOT, 'processed'), exist_ok=True)
        os.makedirs(os.path.join(settings.MEDIA_ROOT, 'masks'), exist_ok=True)
        
        # 3. Background Removal
        processed_filename = f"transparent_{entry.id}.png"
        processed_path = os.path.join(settings.MEDIA_ROOT, 'processed', processed_filename)
        remove_background_hq(original_path, processed_path)
        
        # 4. Clothing Mask
        mask_filename = f"mask_{entry.id}.png"
        mask_path = os.path.join(settings.MEDIA_ROOT, 'masks', mask_filename) 
        generate_clothing_mask(original_path, mask_path)

        # 5. Dominant Colors
        colors = get_dominant_colors(original_path, mask_path) 

# --- AGE/SEX ESTIMATION (FIXED: PIXEL-FIRST MATH) ---
        
        # 1. Get Keypoints directly in PIXELS (not normalized)
        # This prevents the "thin strip" bug on vertical images
        nose_x_px = int(human_data['keypoints'][0]['x'] * w)
        nose_y_px = int(human_data['keypoints'][0]['y'] * h)
        
        r_eye_x_px = int(human_data['keypoints'][3]['x'] * w)
        l_eye_x_px = int(human_data['keypoints'][6]['x'] * w)

        # 2. Calculate Face Size based on Eye Distance (in Pixels)
        eye_dist_px = abs(r_eye_x_px - l_eye_x_px)
        
        # A face is roughly 3x to 4x the width of the eye distance
        face_w_px = int(eye_dist_px * 3.5) 
        face_h_px = int(face_w_px * 1.3) # Faces are usually taller than wide

        # 3. Calculate Center and Start Points
        # The nose is roughly in the center, slightly lower than middle
        x_center = nose_x_px
        y_center = nose_y_px - int(face_h_px * 0.1) # Shift box up slightly

        x_start_px = int(x_center - (face_w_px / 2))
        y_start_px = int(y_center - (face_h_px / 2))

        # 4. CLAMPING (Crucial safety step)
        # Ensure the box doesn't go off the image edge
        x_start_px = max(0, x_start_px)
        y_start_px = max(0, y_start_px)
        
        # Ensure width/height don't exceed image bounds
        real_width = min(face_w_px, w - x_start_px)
        real_height = min(face_h_px, h - y_start_px)
        
        face_bbox_pixels = (x_start_px, y_start_px, real_width, real_height)
        
        # 5. Call Estimation
        estimated_age, estimated_gender, gender_conf = estimate_age_sex(original_path, face_bbox_pixels)

        # --- SAVE ---
        entry.processed_file = os.path.join('processed', processed_filename)
        entry.mask_file = os.path.join('masks', mask_filename) 
        
        entry.metadata = {
            "bbox": human_data['bbox'],
            "keypoints": human_data['keypoints'],
            "detection_confidence": float(human_data['confidence']), # Safety cast
            "dominant_colors": colors,
            "estimated_age": estimated_age,
            "estimated_gender": estimated_gender,
            "gender_confidence": gender_conf # This is now a float from ai_engine
        }
        entry.status = 'COMPLETED'
        entry.save()

    except (IOError, OSError, OperationalError) as e:
        if self.request.retries < self.max_retries:
            entry.error_log = f"Transient error: {e}"
            entry.save()
            raise self.retry(exc=e, countdown=5)
        else:
            entry.status = 'FAILED'; entry.error_log = f"Max retries: {e}"; entry.save()
            
    except Exception as e:
        entry.status = 'FAILED'
        entry.error_log = f"Permanent error: {e}"
        entry.save()