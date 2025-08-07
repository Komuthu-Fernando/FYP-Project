import pytesseract
from PIL import Image
import os

pytesseract.pytesseract.tesseract_cmd = "/opt/homebrew/bin/tesseract"

input_folder = "input_images"
output_folder = "output_text"

os.makedirs(output_folder, exist_ok=True)

def extract_text_from_images():
    for image_file in os.listdir(input_folder):
        if image_file.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')):
            image_path = os.path.join(input_folder, image_file)
            img = Image.open(image_path)
            text = pytesseract.image_to_string(img, lang="sin", config="--psm 6")
            output_file = os.path.join(output_folder, os.path.splitext(image_file)[0] + ".txt")
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(text)
            print(f"Extracted text saved to: {output_file}")

if __name__ == "__main__":
    print("Starting text extraction from images...")
    extract_text_from_images()
    print("Text extraction completed successfully!")
