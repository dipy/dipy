def check_img_shapes(images):
    """
    Checks if the images have same shapes
    """
    base_shape = images[0][0].shape[:3]
    for img in images:
        data, _ = img
        if base_shape != data.shape[:3]:
            return False
    return True
