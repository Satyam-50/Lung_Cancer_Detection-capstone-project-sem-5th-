import numpy as np
import cv2
from skimage import morphology, measure

def normalize_ct(slice_img, min_hu=-1000.0, max_hu=400.0):
    """
    Standard lung window normalization.
    Maps Hounsfield Units [-1000, 400] to [0, 1].
    """
    img = np.clip(slice_img, min_hu, max_hu)
    img = (img - min_hu) / (max_hu - min_hu)
    return img.astype(np.float32)

def extract_lung_roi(img_2d):
    """
    Segments lung tissue from background and chest wall using thresholding and morphological cleaning.
    img_2d: 2D array normalized in [0, 1] or [0, 255]
    Returns: binary lung mask (H, W)
    """
    if img_2d.max() <= 1.0:
        img_uint8 = (img_2d * 255).astype(np.uint8)
    else:
        img_uint8 = img_2d.astype(np.uint8)

    # 1. Binary thresholding (air/lung tissue is darker)
    _, binary = cv2.threshold(img_uint8, 70, 255, cv2.THRESH_BINARY_INV)
    
    # 2. Clear borders to remove external scanner bed/air
    cleared = morphology.clear_border(binary > 0)
    
    # 3. Label connected components
    label_image = measure.label(cleared)
    regions = measure.regionprops(label_image)
    
    if not regions:
        # Fallback mask if thresholding finds no regions
        return np.ones_like(img_2d, dtype=np.uint8)
        
    # Sort regions by area descending
    regions.sort(key=lambda r: r.area, reverse=True)
    
    # Keep the two largest components (left and right lungs)
    lung_mask = np.zeros_like(img_uint8, dtype=np.uint8)
    for r in regions[:2]:
        lung_mask[label_image == r.label] = 1

    # 4. Fill internal holes & morphological closing
    lung_mask = morphology.binary_closing(lung_mask, morphology.disk(5))
    lung_mask = morphology.remove_small_holes(lung_mask, area_threshold=500)
    
    return lung_mask.astype(np.uint8)

def extract_nodule_patches(ct_img, pred_mask, patch_size=(128, 128), min_area=10):
    """
    Finds nodule candidates from predicted mask and crops bounding box patches.
    Returns: list of dicts with patch, bbox (ymin, xmin, ymax, xmax), and area.
    """
    if pred_mask.max() <= 1.0:
        mask_uint8 = (pred_mask > 0.5).astype(np.uint8) * 255
    else:
        mask_uint8 = (pred_mask > 127).astype(np.uint8) * 255
        
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    patches = []
    H, W = ct_img.shape[:2]
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
            
        x, y, w, h = cv2.boundingRect(cnt)
        # Find center of nodule
        cx, cy = x + w // 2, y + h // 2
        
        # Calculate square patch bounding box
        half_w = max(w, h, patch_size[0] // 2)
        xmin = max(0, cx - half_w)
        ymin = max(0, cy - half_w)
        xmax = min(W, cx + half_w)
        ymax = min(H, cy + half_w)
        
        patch = ct_img[ymin:ymax, xmin:xmax]
        patch_resized = cv2.resize(patch, patch_size, interpolation=cv2.INTER_CUBIC)
        
        # Convert to 3-channel if grayscale
        if len(patch_resized.shape) == 2:
            patch_rgb = cv2.cvtColor(patch_resized.astype(np.float32), cv2.COLOR_GRAY2RGB)
        else:
            patch_rgb = patch_resized
            
        patches.append({
            "patch": patch_rgb,
            "bbox": (ymin, xmin, ymax, xmax),
            "area": area,
            "center": (cx, cy)
        })
        
    return patches

def generate_sample_ct(sample_type="malignant"):
    """
    Generates high-fidelity anatomical CT scan slice for interactive testing & demonstration.
    sample_type: 'malignant', 'benign', or 'normal'
    """
    H, W = 512, 512
    img = np.zeros((H, W), dtype=np.float32)
    
    # Draw body contour / chest wall
    cv2.ellipse(img, (256, 256), (220, 200), 0, 0, 360, 0.7, -1)
    
    # Left and Right Lung Cavities
    cv2.ellipse(img, (180, 250), (60, 110), 10, 0, 360, 0.15, -1)
    cv2.ellipse(img, (332, 250), (60, 110), -10, 0, 360, 0.15, -1)
    
    # Mediastinum and Spine
    cv2.circle(img, (256, 380), 25, 0.9, -1) # Spine
    cv2.ellipse(img, (256, 260), (35, 75), 0, 0, 360, 0.8, -1) # Heart/Mediastinum
    
    # Add subtle lung pulmonary vascular markings
    for i in range(12):
        pt1 = (180, 250)
        pt2 = (int(180 + 35 * np.cos(i)), int(250 + 70 * np.sin(i)))
        cv2.line(img, pt1, pt2, 0.25, 1)
        
        pt3 = (332, 250)
        pt4 = (int(332 + 35 * np.cos(i)), int(250 + 70 * np.sin(i)))
        cv2.line(img, pt3, pt4, 0.25, 1)
        
    mask = np.zeros((H, W), dtype=np.float32)
    
    if sample_type == "malignant":
        # Spiculated, irregular malignant lesion in right upper/mid lobe
        cx, cy = 345, 230
        cv2.circle(img, (cx, cy), 14, 0.75, -1)
        # Spiculations
        for angle in np.linspace(0, 2*np.pi, 8, endpoint=False):
            ex = int(cx + 20 * np.cos(angle))
            ey = int(cy + 20 * np.sin(angle))
            cv2.line(img, (cx, cy), (ex, ey), 0.7, 2)
            cv2.line(mask, (cx, cy), (ex, ey), 1.0, 2)
        cv2.circle(mask, (cx, cy), 14, 1.0, -1)
        
    elif sample_type == "benign":
        # Smooth, well-circumscribed round benign nodule in left lung
        cx, cy = 175, 270
        cv2.circle(img, (cx, cy), 9, 0.65, -1)
        cv2.circle(mask, (cx, cy), 9, 1.0, -1)
        
    # Add Gaussian noise
    noise = np.random.normal(0, 0.02, (H, W))
    img = np.clip(img + noise, 0, 1.0)
    
    return img.astype(np.float32), mask.astype(np.float32)
