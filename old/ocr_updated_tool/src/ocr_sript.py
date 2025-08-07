import os
import cv2
import pytesseract
from datetime import datetime

# Define folder paths dynamically
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
input_folder = os.path.join(BASE_DIR, "data", "untitled folder")
output_folder = os.path.join(BASE_DIR, "data", "processed_images")
csv_output = os.path.join(BASE_DIR, "data", "ocr_results1.csv")

# Ensure required folders exist
os.makedirs(output_folder, exist_ok=True)

# Initialize CSV file with headers if it doesn't exist
if not os.path.exists(csv_output):
    with open(csv_output, "w", encoding="utf-8") as f:
        f.write("")

# Process each image in the input folder
for filename in os.listdir(input_folder):
    if filename.endswith((".png", ".jpg", ".jpeg")):
        image_path = os.path.join(input_folder, filename)
        image = cv2.imread(image_path)

        # Convert to Grayscale and Apply Thresholding
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)

        # Perform OCR
        custom_config = r'--oem 3 --psm 6'
        data = pytesseract.image_to_data(thresh, config=custom_config, lang="sin", output_type=pytesseract.Output.DICT)

        extracted_lines = []
        current_line = []

        # Process Detected Words Line-wise
        for i in range(len(data["text"])):
            word = data["text"][i].strip()
            if word:
                current_line.append(word)
                if word.endswith(")"):  # End of line indicator
                    extracted_lines.append(" ".join(current_line))
                    current_line = []

        # Save results to CSV line by line
        if extracted_lines:
            with open(csv_output, "a", encoding="utf-8") as f:
                f.write(f"{filename}\n")  # Image name
                for line in extracted_lines:
                    f.write(f"{line}\n")  # Each extracted line
                f.write("\n")  # Empty line to separate different images

        # Save the Annotated Image
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename_no_ext = os.path.splitext(filename)[0]
        output_filename = f"annotated_{filename_no_ext}_{timestamp}.png"
        output_path = os.path.join(output_folder, output_filename)
        cv2.imwrite(output_path, image)

        print(f"✅ Processed {filename}, saved annotated image and updated CSV.")

print(f"🎉 Processing Complete! Results saved to {csv_output}")
