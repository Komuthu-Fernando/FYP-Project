import os
import pytesseract
from PIL import Image

# Set the path to the Tesseract executable (default for Homebrew installation on Mac)
pytesseract.pytesseract.tesseract_cmd = r'/opt/homebrew/bin/tesseract'  # Update if your path differs

def extract_sinhala_text(image_path, output_path):
    """
    Extract Sinhala text from an image and save it to a text file.
    
    Args:
        image_path (str): Path to the input image.
        output_path (str): Path to the output text file.
    """
    try:
        # Verify that the image file exists
        if not os.path.isfile(image_path):
            raise FileNotFoundError(f"Image file not found at: {image_path}")

        # Open the image
        with Image.open(image_path) as img:
            # Perform OCR with Sinhala language support and preserve line structure
            # --psm 6 assumes a single uniform block of text, aiding line-by-line extraction
            text = pytesseract.image_to_string(img, lang='sin', config='--psm 6')
        
        # Ensure the output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save the extracted text to the output file with UTF-8 encoding
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(text)
        
        print(f"Text successfully extracted and saved to: {output_path}")
    
    except FileNotFoundError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

def main():
    # Define input and output folders (update these paths as needed)
    input_folder = '/Users/ajithfernando/Documents/Komuthu Documents/FYP/Research/Project/ocr_new_tool/input_folder'  # Replace with your input folder path
    output_folder = '/Users/ajithfernando/Documents/Komuthu Documents/FYP/Research/Project/ocr_new_tool/output_folder'  # Replace with your output folder path
    image_file = 'IMG_1.jpg'  # Replace with your image file name

    # Construct full file paths
    image_path = os.path.join(input_folder, image_file)
    output_path = os.path.join(output_folder, 'extracted_text.txt')

    # Extract and save the text
    extract_sinhala_text(image_path, output_path)

if __name__ == "__main__":
    main()