import torch

def center_crop(tensor, target_h, target_w=None):
    """
    Center crop a tensor of shape [N, H, W, C] to [N, target_h, target_w, C].
    
    Args:
        tensor: Input tensor of shape [N, H, W, C]
        target_h: Target height
        target_w: Target width (if None, uses target_h for square crop)
    
    Returns:
        Cropped tensor of shape [N, target_h, target_w, C]
    """
    if target_w is None:
        target_w = target_h
    
    n, h, w, c = tensor.shape
    
    # Calculate crop start positions (centered)
    start_h = (h - target_h) // 2
    start_w = (w - target_w) // 2
    
    # Ensure we don't go out of bounds
    start_h = max(0, start_h)
    start_w = max(0, start_w)
    
    # Perform the crop
    cropped = tensor[:, start_h:start_h + target_h, start_w:start_w + target_w, :]
    
    return cropped

