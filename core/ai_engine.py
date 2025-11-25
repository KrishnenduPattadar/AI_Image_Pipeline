import cv2
import os
import numpy as np
from PIL import Image
from rembg import remove
from sklearn.cluster import KMeans
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from sklearn.cluster import KMeans


# --- ADD THESE LINES TO DEFINE PATHS ---
# This finds the folder where manage.py lives
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Define the exact paths to your downloaded models
POSE_MODEL_PATH = os.path.join(BASE_DIR, 'pose_landmarker_heavy.task')
SEG_MODEL_PATH = os.path.join(BASE_DIR, 'selfie_multiclass_256x256.tflite')

# --- AGE/GENDER MODEL PATHS (Assuming files are in project root) ---
AGE_MODEL_PATH = os.path.join(BASE_DIR, 'age_net.caffemodel')
GENDER_MODEL_PATH = os.path.join(BASE_DIR, 'gender_net.caffemodel')
AGE_PROTO_PATH = os.path.join(BASE_DIR, 'age_deploy.prototxt') # Configuration file
GENDER_PROTO_PATH = os.path.join(BASE_DIR, 'gender_deploy.prototxt')

# Define classification labels
GENDER_LIST = ['Male', 'Female']
AGE_LIST = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']

# Load the models using OpenCV's DNN
AGE_NET = cv2.dnn.readNet(AGE_MODEL_PATH, AGE_PROTO_PATH)
GENDER_NET = cv2.dnn.readNet(GENDER_MODEL_PATH, GENDER_PROTO_PATH)



# --- GOAL 1: VALIDATE IMAGE ---
def validate_image(image_path):
    try:
        with Image.open(image_path) as img:
            img.verify() # Checks for corruption
            if img.format not in ['JPEG', 'PNG', 'WEBP']:
                return False, "Invalid format"
            if img.width < 100 or img.height < 100:
                return False, "Image too small"
        return True, "Valid"
    except Exception:
        return False, "File is not a readable image"

# --- GOAL 2 & 4: DETECT HUMAN + EXTRACT POSE ---
def analyze_human_pose(image_path):
    # Initialize MediaPipe Pose
    base_options = python.BaseOptions(model_asset_path=POSE_MODEL_PATH)
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        output_segmentation_masks=False,
        min_pose_detection_confidence=0.5 # REJECT if < 50%
    )
    
    detector = vision.PoseLandmarker.create_from_options(options)
    image = mp.Image.create_from_file(image_path)
    detection_result = detector.detect(image)

    # REJECT if no body found
    if not detection_result.pose_landmarks:
        return None 

# The first pose found is at index 0. The first landmark (NOSE) is at index 0.
    # The 'visibility' score acts as the confidence measure for that keypoint.
    confidence_score = detection_result.pose_landmarks[0][0].visibility

    # Extract Bounding Box from landmarks
    landmarks = detection_result.pose_landmarks[0]
    x_vals = [lm.x for lm in landmarks]
    y_vals = [lm.y for lm in landmarks]
    
    bbox = {
        "x_min": min(x_vals), "y_min": min(y_vals),
        "x_max": max(x_vals), "y_max": max(y_vals)
    }
    
    # Return serializable keypoints
    keypoints = [{"x": lm.x, "y": lm.y, "z": lm.z} for lm in landmarks]
    
    return {"bbox": bbox, "keypoints": keypoints, "confidence": confidence_score}

# --- GOAL 3: REMOVE BACKGROUND ---
def remove_background_hq(input_path, output_path):
    with open(input_path, 'rb') as i:
        with open(output_path, 'wb') as o:
            input_data = i.read()
            output_data = remove(input_data) # Uses U-2-Net
            o.write(output_data)
    return output_path

# --- GOAL: DOMINANT COLORS (REFINED VERSION) ---
def get_dominant_colors(image_path, mask_path, k=3): # <-- Requires mask_path
    """Uses K-Means clustering to find K dominant colors within the mask."""
    import cv2
    import numpy as np
    from sklearn.cluster import KMeans
    
    img = cv2.imread(image_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE) # Load mask in grayscale
    
    if img is None or mask is None:
        return []

    # 1. Resize mask to match image (Crucial step if they don't match)
    # Note: Using cv2.INTER_NEAREST ensures clean mask resizing
    mask = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)

    # 2. Apply the mask to isolate clothing pixels
    filtered_img = cv2.bitwise_and(img, img, mask=mask)

    # 3. Flatten only the NON-BLACK/NON-ZERO pixels (i.e., inside the clothing)
    img_rgb = cv2.cvtColor(filtered_img, cv2.COLOR_BGR2RGB)
    
    # We filter out the black pixels (which were the background/skin)
    # The mask contains only 0 (background) and 255 (clothing)
    non_zero_pixels = img_rgb[mask != 0]

    # Check if we have enough pixels to cluster
    if non_zero_pixels.shape[0] < k:
        return ["#FFFFFF"] * k # Return white if clothes are too small to cluster

    # 4. Run K-Means Clustering on the filtered pixels
    kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
    kmeans.fit(non_zero_pixels)
    
    # Convert RGB centers to Hex codes
    colors = []
    for center in kmeans.cluster_centers_:
        colors.append('#%02x%02x%02x' % (int(center[0]), int(center[1]), int(center[2])))
        
    return colors

# --- GOAL 6: CLOTHING MASK (Coarse) ---
def generate_clothing_mask(image_path, output_mask_path):
    # Uses MediaPipe Multi-class Segmenter
    # Class 4 = Clothes
    base_options = python.BaseOptions(model_asset_path=SEG_MODEL_PATH)
    options = vision.ImageSegmenterOptions(base_options=base_options, output_category_mask=True)
    
    segmenter = vision.ImageSegmenter.create_from_options(options)
    image = mp.Image.create_from_file(image_path)
    segmentation_result = segmenter.segment(image)
    
    category_mask = segmentation_result.category_mask.numpy_view()
    
    # Create binary mask where pixel == 4 (Clothes)
    clothing_mask = np.where(category_mask == 4, 255, 0).astype(np.uint8)
    
    cv2.imwrite(output_mask_path, clothing_mask)
    return output_mask_path



#This function takes the image and the face location, then uses OpenCV's DNN module to classify.
def estimate_age_sex(image_path, face_bbox_px): # <-- Renamed input to clarify it's in PIXELS
    """Crops the face using pixel coordinates and predicts age/gender."""
    
    img = cv2.imread(image_path)
    if img is None: return None, None
    
    h_img, w_img, _ = img.shape # Get actual image dimensions
    
    # Unpack the pixel BBox from tasks.py
    x, y, w, h = face_bbox_px 
    
    # 1. Apply a small buffer/padding (in pixels)
    padding = 10 
    
    # 2. Calculate SAFE crop coordinates (Clamping to prevent going outside the image)
    x1 = max(0, x - padding)
    y1 = max(0, y - padding)
    x2 = min(w_img, x + w + padding)
    y2 = min(h_img, y + h + padding)
    
    # 3. Crop the face image
    face_img = img[y1:y2, x1:x2]
    
    # Check if the cropped area is still valid
    if face_img.size == 0:
        return "N/A", "N/A" # Cannot crop face, skip estimation

    # 4. Prepare the input blob (standard CV step)
    # The image is resized to 227x227 for the Caffe model
    blob = cv2.dnn.blobFromImage(face_img, 1.0, (227, 227), (78.4263377603, 87.7689143744, 114.895847746), swapRB=False)
    
    # 5. Gender Prediction
    GENDER_NET.setInput(blob)
    gender_preds = GENDER_NET.forward()
    gender = GENDER_LIST[gender_preds[0].argmax()]
    
    # 6. Age Prediction
    AGE_NET.setInput(blob)
    age_preds = AGE_NET.forward()
    age = AGE_LIST[age_preds[0].argmax()]
    
    return age, gender



def get_image_dimensions(image_path):
    """Safely reads the image to return its width and height in pixels."""
    import cv2
    img = cv2.imread(image_path)
    
    if img is None:
        # If OpenCV can't read the image, return zero dimensions
        return 0, 0 
    
    # cv2.imread returns array shape: (Height, Width, Channels). 
    # We return (Width, Height).
    return img.shape[1], img.shape[0]