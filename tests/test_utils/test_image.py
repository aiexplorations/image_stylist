import pytest
from PIL import Image
import numpy as np
from src.utils.image import (
    decode_base64_image,
    encode_pil_to_base64,
    prepare_image_for_model,
    process_generated_image
)

def test_decode_base64_image(sample_base64_image):
    """Test decoding base64 image string to PIL Image."""
    image = decode_base64_image(sample_base64_image)
    assert isinstance(image, Image.Image)
    assert image.mode == "RGB"
    assert image.size == (512, 512)

def test_decode_base64_image_invalid():
    """Test decoding invalid base64 string."""
    with pytest.raises(ValueError):
        decode_base64_image("invalid_base64")

def test_encode_pil_to_base64(sample_image):
    """Test encoding PIL Image to base64 string."""
    base64_str = encode_pil_to_base64(sample_image)
    assert isinstance(base64_str, str)
    assert base64_str.startswith("data:image/jpeg;base64,")
    
    # Verify we can decode it back
    decoded = decode_base64_image(base64_str)
    assert isinstance(decoded, Image.Image)

def test_prepare_image_for_model(sample_image):
    """Test image preparation for model input."""
    # Test with default size
    prepared = prepare_image_for_model(sample_image)
    assert isinstance(prepared, Image.Image)
    assert prepared.size == (512, 512)
    
    # Test with custom size
    custom_size = (256, 256)
    prepared = prepare_image_for_model(sample_image, custom_size)
    assert prepared.size == custom_size

def test_prepare_image_for_model_aspect_ratio():
    """Test aspect ratio preservation in image preparation."""
    # Create a wide image
    wide_image = Image.new('RGB', (800, 400))
    prepared = prepare_image_for_model(wide_image)
    
    # The image should be padded to maintain aspect ratio
    assert prepared.size == (512, 512)
    
    # Create a tall image
    tall_image = Image.new('RGB', (400, 800))
    prepared = prepare_image_for_model(tall_image)
    assert prepared.size == (512, 512)

def test_process_generated_image():
    """Test processing of generated images."""
    # Test with PIL Image
    input_image = Image.new('RGB', (512, 512), color='red')
    processed = process_generated_image(input_image)
    assert isinstance(processed, Image.Image)
    assert processed.size == (512, 512)
    
    # Test with numpy array
    np_image = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
    processed = process_generated_image(np_image)
    assert isinstance(processed, Image.Image)
    assert processed.size == (512, 512)
    
    # Test with custom output size
    processed = process_generated_image(input_image, output_size=(256, 256))
    assert processed.size == (256, 256) 