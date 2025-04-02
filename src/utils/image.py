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
    output_size: Optional[Tuple[int, int]] = None,
    remove_black_bands: bool = True,
    min_dimension: int = 1280
) -> Image.Image:
    """
    Process a generated image for output.
    
    Args:
        image: Generated image (PIL Image or numpy array)
        output_size: Optional target size for resizing
        remove_black_bands: Whether to remove black bands from the image
        min_dimension: Minimum size for the largest dimension
        
    Returns:
        PIL.Image: Processed output image
    """
    if isinstance(image, np.ndarray):
        image = Image.fromarray(np.uint8(image))
    
    if remove_black_bands:
        image = remove_black_bands_from_image(image)
    
    # Resize to ensure minimum dimension
    if min_dimension > 0:
        current_width, current_height = image.size
        max_dimension = max(current_width, current_height)
        if max_dimension < min_dimension:
            # Calculate scale factor to reach the minimum dimension
            scale_factor = min_dimension / max_dimension
            new_width = int(current_width * scale_factor)
            new_height = int(current_height * scale_factor)
            image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
    # Apply specific output size if provided (overrides min_dimension)
    elif output_size:
        image = image.resize(output_size, Image.Resampling.LANCZOS)
    
    return image

def remove_black_bands_from_image(image: Image.Image) -> Image.Image:
    """
    Removes black bands (padding) from an image by cropping to content.
    
    Args:
        image: PIL Image to process
        
    Returns:
        PIL.Image: Image with black bands removed
    """
    # Convert to numpy array for processing
    img_array = np.array(image)
    
    # Check if the image has 3 channels (RGB)
    if len(img_array.shape) == 3 and img_array.shape[2] == 3:
        # Convert to grayscale for easier analysis
        gray = np.mean(img_array, axis=2)
    else:
        # Already grayscale
        gray = img_array
    
    # Find rows and columns that are not black (threshold > 10 to handle compression artifacts)
    rows = np.any(gray > 10, axis=1)
    cols = np.any(gray > 10, axis=0)
    
    # Find the boundaries of content
    y_min, y_max = np.where(rows)[0][[0, -1]] if np.any(rows) else (0, image.height-1)
    x_min, x_max = np.where(cols)[0][[0, -1]] if np.any(cols) else (0, image.width-1)
    
    # Add a small padding (1% of dimension) to avoid cutting content too tight
    padding_y = max(1, int(image.height * 0.01))
    padding_x = max(1, int(image.width * 0.01))
    
    y_min = max(0, y_min - padding_y)
    y_max = min(image.height - 1, y_max + padding_y)
    x_min = max(0, x_min - padding_x)
    x_max = min(image.width - 1, x_max + padding_x)
    
    # Crop the image to content
    return image.crop((x_min, y_min, x_max + 1, y_max + 1)) 