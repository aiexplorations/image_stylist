from typing import Tuple, Union, Optional
import base64
from io import BytesIO
from PIL import Image
import numpy as np

def decode_base64_image(base64_string: str) -> Image.Image:
    """
    Decode a base64 string into a PIL Image.
    
    Args:
        base64_string: Base64 encoded image string
        
    Returns:
        PIL.Image: Decoded image
        
    Raises:
        ValueError: If the base64 string is invalid
    """
    try:
        # Remove data URL prefix if present
        if ',' in base64_string:
            base64_string = base64_string.split(',')[1]
            
        image_data = base64.b64decode(base64_string)
        image = Image.open(BytesIO(image_data))
        
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
            
        return image
    except Exception as e:
        raise ValueError(f"Invalid base64 image: {str(e)}")

def encode_pil_to_base64(image: Image.Image) -> str:
    """
    Encode a PIL image to base64 string.
    
    Args:
        image: PIL Image to encode
        
    Returns:
        str: Base64 encoded image string with data URL prefix
    """
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return f"data:image/jpeg;base64,{img_str}"

def prepare_image_for_model(
    image: Image.Image,
    target_size: Tuple[int, int] = (512, 512)
) -> Image.Image:
    """
    Prepare an image for model input by resizing and normalizing.
    
    Args:
        image: Input PIL Image
        target_size: Desired output size (width, height)
        
    Returns:
        PIL.Image: Processed image ready for model input
    """
    # Resize image while maintaining aspect ratio
    ratio = min(target_size[0] / image.size[0], target_size[1] / image.size[1])
    new_size = tuple(int(dim * ratio) for dim in image.size)
    image = image.resize(new_size, Image.Resampling.LANCZOS)
    
    # Create new image with padding
    new_image = Image.new("RGB", target_size, (0, 0, 0))
    paste_pos = ((target_size[0] - new_size[0]) // 2,
                 (target_size[1] - new_size[1]) // 2)
    new_image.paste(image, paste_pos)
    
    return new_image

def process_generated_image(
    image: Union[Image.Image, np.ndarray],
    output_size: Optional[Tuple[int, int]] = None
) -> Image.Image:
    """
    Process a generated image for output.
    
    Args:
        image: Generated image (PIL Image or numpy array)
        output_size: Optional target size for resizing
        
    Returns:
        PIL.Image: Processed output image
    """
    if isinstance(image, np.ndarray):
        image = Image.fromarray(np.uint8(image))
    
    if output_size:
        image = image.resize(output_size, Image.Resampling.LANCZOS)
    
    return image 