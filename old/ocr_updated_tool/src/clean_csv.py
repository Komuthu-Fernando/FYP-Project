import os
import re

# Define file paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
csv_input = os.path.join(BASE_DIR, "data", "ocr_results.csv")
csv_output = os.path.join(BASE_DIR, "data", "ocr_results_cleaned.csv")

# Sinhala Unicode Range
SINHALA_PATTERN = r"[\u0D80-\u0DFF]+"  # Matches only Sinhala characters

def clean_text(line):
    """ Remove symbols and numbers, keeping only Sinhala text and spaces. """
    words = re.findall(SINHALA_PATTERN, line)  # Extract only Sinhala words
    return " ".join(words) if words else ""  # Join words back with spaces

# Process the CSV file
with open(csv_input, "r", encoding="utf-8") as infile, open(csv_output, "w", encoding="utf-8") as outfile:
    for line in infile:
        cleaned_line = clean_text(line.strip())
        if cleaned_line or "," not in line:  # Keep image names, remove empty lines
            outfile.write(cleaned_line + "\n")

print(f"✅ Cleaning Complete! Cleaned CSV saved as {csv_output}")
