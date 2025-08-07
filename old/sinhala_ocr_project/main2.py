import pytesseract
from PIL import Image, ImageEnhance, ImageFilter
import os
import re

pytesseract.pytesseract.tesseract_cmd = "/opt/homebrew/bin/tesseract"

input_folder = "input_images"
output_folder = "output_text"

os.makedirs(output_folder, exist_ok=True)

def preprocess_image(image):
    image = image.convert('L')
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(2)
    image = image.point(lambda x: 0 if x < 128 else 255)
    image = image.filter(ImageFilter.GaussianBlur(radius=1))
    return image

def filter_sinhala_text(text):
    sinhala_pattern = re.compile(r'[\u0D80-\u0DFF=]+')
    filtered_lines = []
    for line in text.splitlines():
        sinhala_line = ' '.join(sinhala_pattern.findall(line))
        if sinhala_line:
            filtered_lines.append(sinhala_line)
    return '\n'.join(filtered_lines)

def extract_text_from_images():
    for image_file in os.listdir(input_folder):
        if image_file.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')):
            image_path = os.path.join(input_folder, image_file)
            img = Image.open(image_path)
            img = preprocess_image(img)
            text = pytesseract.image_to_string(img, lang="sin", config="--psm 6")
            filtered_text = filter_sinhala_text(text)
            output_file = os.path.join(output_folder, os.path.splitext(image_file)[0] + ".txt")
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(filtered_text)
            print(f"Extracted text saved to: {output_file}")

if __name__ == "__main__":
    print("Starting text extraction from images...")
    extract_text_from_images()
    print("Text extraction completed successfully!")
