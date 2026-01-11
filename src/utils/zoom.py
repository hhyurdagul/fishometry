import cv2
import numpy as np

def augment_zoom(image, type="in", magnitude=0.1):
    h, w = image.shape[:2]
    
    # Validation
    magnitude = max(0.0, min(0.9, magnitude)) # Clamp to safe range
    ratio = 1.0 - magnitude
    
    if type == "in":
        # Zoom in: Crop center region of size (h*ratio, w*ratio) and resize to (h, w)
        # Ratio 0.8 means we keep 80% of the image (0.2 magnitude zoom)
        nh, nw = int(h * ratio), int(w * ratio)
        
        # Top left corner
        y = (h - nh) // 2
        x = (w - nw) // 2
        
        cropped = image[y:y+nh, x:x+nw]
        resized = cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)
        return resized
        
    elif type == "out":
        # Zoom out: Simply resize image to (h*ratio, w*ratio)
        # Ratio 0.8 means result is 80% of original size (0.2 magnitude zoom)
        nh, nw = int(h * ratio), int(w * ratio)
        resized = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR)
        return resized
    
    return image
