import numpy as np

palette = [
    [0,0,0],
    [255, 153, 51], # orange (double white solid) (old double white)
    [255, 51,255], # fucsia (old other) (double white dashed)
    [0, 255, 0], # green (single white solid)
    [255,0,0], # red (single white dashed)
    [190, 189, 127], # dirty green (single yellow solid)
    [116,18,29], # dark red (single yellow dashed)
    [149, 95, 32], # brown (double yellow dashed)
    [184, 183, 153], # beige (double yellow solid)
    [0,0,255], # blue (crosswalk)
    [68, 17,81], # purple (curb)
    [255, 255, 255], # white (drivable)
    # TODO: pick new
    [61,82,213], # light blue
    [0,79,45], # dark green
    [181,141,182], # dirty pink
    [108, 70, 117], # dirty fucsia
    [190, 189, 127], # dirty green
    [255, 164, 32], # orange
    [49, 127, 67], # green   
]


def map2D_to_RGB(img):
    """Convert 2d integer label vector to an rbg image with the same size

    Args:
        img: prediction to convert

    Returns:
        rgb image rpresenting classes
    """
    output = np.ones((*img.shape, 3), dtype=np.uint8)
    classes = np.unique(img.astype(np.uint8))
    for c in classes:
        output[img == c] = np.array(palette[c], dtype=np.uint8)
    
    return output