from PIL import Image
import os

def resize_image(input_path, output_path, new_width, new_height):
    """
    Resize an image to the specified dimensions.

    :param input_path: Path to the input image
    :param output_path: Path to save the resized image
    :param new_width: Desired width of the output image
    :param new_height: Desired height of the output image
    """
    # Open the image
    with Image.open(input_path) as img:
        img_resized = img.resize((new_width, new_height), Image.LANCZOS)
        img_resized.save(output_path)     
    print(f"Image saved to {output_path}")

# Example Usage
input_image = "Images/image5.jpg"   # Change this to your image path
output_image = "Images/image5.jpg" # Output image name
new_width = 700  # Desired width
new_height = 600  # Desired height

resize_image(input_image, output_image, new_width, new_height)
