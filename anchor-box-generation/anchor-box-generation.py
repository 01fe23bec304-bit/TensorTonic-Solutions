import numpy as np

def generate_anchors(feature_size, image_size, scales, aspect_ratios):
    """
    Generate anchor boxes for object detection.
    
    Returns:
        anchors: list of [x1, y1, x2, y2]
    """
    
    anchors = []
    
    # stride between grid centers
    stride = image_size / feature_size
    
    for i in range(feature_size):
        for j in range(feature_size):
            
            # center coordinates
            cx = (j + 0.5) * stride
            cy = (i + 0.5) * stride
            
            for scale in scales:
                for ratio in aspect_ratios:
                    
                    # compute width and height
                    w = scale * np.sqrt(ratio)
                    h = scale / np.sqrt(ratio)
                    
                    # anchor box corners
                    x1 = cx - w / 2
                    y1 = cy - h / 2
                    x2 = cx + w / 2
                    y2 = cy + h / 2
                    
                    anchors.append([x1, y1, x2, y2])
    
    return anchors