#!/usr/bin/env python3
"""
Test script to verify base64 encoding/decoding for images.
This helps debug issues with image data transmission.
"""

import sys
import base64
from io import BytesIO
from PIL import Image

def encode_image(input_path, output_path=None):
    """Encode an image to base64 and optionally save it"""
    try:
        # Open the image
        with Image.open(input_path) as img:
            print(f"Opened image: {img.format}, size: {img.size}, mode: {img.mode}")
            
            # Convert to RGB if needed
            if img.mode != 'RGB':
                img = img.convert('RGB')
                print(f"Converted to RGB mode")
            
            # Encode to base64
            buffered = BytesIO()
            img.save(buffered, format="PNG")
            img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
            data_url = f"data:image/png;base64,{img_base64}"
            
            print(f"Base64 string length: {len(img_base64)}")
            print(f"Data URL length: {len(data_url)}")
            
            # Optionally save the data URL to a file
            if output_path:
                with open(output_path, 'w') as f:
                    f.write(data_url)
                print(f"Saved data URL to {output_path}")
            
            return data_url
    except Exception as e:
        print(f"Error encoding image: {e}")
        return None

def decode_image(base64_string, output_path):
    """Decode a base64 string back to an image and save it"""
    try:
        # Remove the data URL prefix if present
        if base64_string.startswith('data:'):
            base64_string = base64_string.split(',')[1]
        
        # Decode the base64 string
        img_data = base64.b64decode(base64_string)
        print(f"Decoded base64 data length: {len(img_data)} bytes")
        
        # Create an image from the decoded data
        img = Image.open(BytesIO(img_data))
        print(f"Created image: {img.format}, size: {img.size}, mode: {img.mode}")
        
        # Save the image
        img.save(output_path)
        print(f"Saved decoded image to {output_path}")
        
        return True
    except Exception as e:
        print(f"Error decoding image: {e}")
        return False

def test_roundtrip(input_path, temp_file_path, output_path):
    """Test the full round-trip process: image → base64 → image"""
    # Encode
    data_url = encode_image(input_path, temp_file_path)
    if not data_url:
        return False
    
    # Decode
    return decode_image(data_url, output_path)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <input_image_path> [output_image_path]")
        sys.exit(1)
    
    input_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else "output_test.png"
    temp_file_path = "temp_base64.txt"
    
    print(f"Testing base64 encoding/decoding on {input_path}")
    if test_roundtrip(input_path, temp_file_path, output_path):
        print("✅ Test successful! Base64 encoding and decoding worked correctly.")
    else:
        print("❌ Test failed. See above errors for details.")
