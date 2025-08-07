# import pandas as pd

# # File paths
# filtered_file = "filtered_inscriptions.csv"
# brahmi_dic_file = "brahmi_dictionary 11 (1).csv"

# # Load the files
# filtered_df = pd.read_csv(filtered_file)
# brahmi_df = pd.read_csv(brahmi_dic_file, header=None)  # No headers assumed

# # Extract the Sinhala words from both
# filtered_words = set()
# for sentence in filtered_df["අක්ෂර පරිවර්තනය"].dropna():
#     words = str(sentence).strip().split()
#     filtered_words.update(words)

# brahmi_words = set(brahmi_df[0].dropna().astype(str).str.strip())

# # Find common and unique words
# common_words = sorted(filtered_words.intersection(brahmi_words))
# extra_brahmi_words = sorted(brahmi_words.difference(filtered_words))

# # Save to CSV files
# pd.DataFrame(common_words, columns=["common_words"]).to_csv("common_words.csv", index=False)
# pd.DataFrame(extra_brahmi_words, columns=["extra_brahmi_words"]).to_csv("extra_brahmi_words.csv", index=False)

# print("✅ Files created: common_words.csv and extra_brahmi_words.csv")

# import pandas as pd
# import re

# # File paths
# filtered_file = "filtered_inscriptions.csv"
# brahmi_dic_file = "brahmi_dictionary 11 (1).csv"
# output_file = "word_comparison_output.csv"

# # Load CSVs
# filtered_df = pd.read_csv(filtered_file)
# brahmi_df = pd.read_csv(brahmi_dic_file, header=None)

# # Set of Brahmi dictionary words
# brahmi_words = set(brahmi_df[0].dropna().astype(str).str.strip())

# # Extract Sinhala words from inscriptions (remove punctuation and numbers)
# def extract_words(text):
#     if pd.isna(text):
#         return []
#     # Remove punctuation, numbers, and split by whitespace
#     words = re.findall(r'[\u0D80-\u0DFF]+', text)
#     return words

# filtered_words = set()

# for text in filtered_df["අක්ෂර පරිවර්තනය"]:
#     words = extract_words(text)
#     filtered_words.update(words)

# # Find matching, extra (filtered only), and missing (brahmi only)
# common_words = filtered_words & brahmi_words
# extra_in_filtered = filtered_words - brahmi_words
# missing_in_filtered = brahmi_words - filtered_words

# # Create final sorted list with label
# results = []

# for word in sorted(common_words):
#     results.append((word, "In Both"))

# for word in sorted(extra_in_filtered):
#     results.append((word, "Only in Filtered"))

# for word in sorted(missing_in_filtered):
#     results.append((word, "Only in Brahmi Dic"))

# # Save to CSV with UTF-8 BOM for Sinhala readability in Excel
# output_df = pd.DataFrame(results, columns=["Word", "Source"])
# output_df.to_csv(output_file, index=False, encoding="utf-8-sig")

# print("Comparison saved to:", output_file)

import pandas as pd
import re

# File paths
filtered_file = "filtered_inscriptions.csv"
brahmi_dic_file = "brahmi_dictionary 11 (1).csv"
output_file = "word_comparison_output.csv"

# Load CSVs
filtered_df = pd.read_csv(filtered_file)
brahmi_df = pd.read_csv(brahmi_dic_file, header=None, names=["Word", "Meaning"])

# Normalize Brahmi dictionary words
brahmi_df["Word"] = brahmi_df["Word"].astype(str).str.strip()
brahmi_dict = dict(zip(brahmi_df["Word"], brahmi_df["Meaning"].fillna("")))
brahmi_words = set(brahmi_dict.keys())

# Extract Sinhala words
def extract_words(text):
    if pd.isna(text):
        return []
    return re.findall(r'[\u0D80-\u0DFF]+', text)

filtered_words = set()

for text in filtered_df["අක්ෂර පරිවර්තනය"]:
    words = extract_words(text)
    filtered_words.update(words)

# Word groups
common_words = filtered_words & brahmi_words
only_in_filtered = filtered_words - brahmi_words
only_in_brahmi = brahmi_words - filtered_words

# Collect results
results = []

for word in sorted(common_words):
    results.append((word, "In Both", brahmi_dict.get(word, "")))

for word in sorted(only_in_filtered):
    results.append((word, "Only in Filtered", ""))

for word in sorted(only_in_brahmi):
    results.append((word, "Only in Brahmi Dic", brahmi_dict.get(word, "")))

# Create DataFrame and save
output_df = pd.DataFrame(results, columns=["Word", "Source", "Meaning"])
output_df.to_csv(output_file, index=False, encoding="utf-8-sig")

print(f"Word comparison with meanings saved to: {output_file}")
