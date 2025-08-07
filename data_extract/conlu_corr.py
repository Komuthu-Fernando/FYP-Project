import pandas as pd

# Load the Excel file
input_file = "input.xlsx"
output_file = "output_fixed.xlsx"

# Read the entire sheet without headers (we'll handle them manually)
df = pd.read_excel(input_file, header=None, dtype=str)

# Define expected columns
columns = ['ID', 'FORM', 'LEMMA', 'UPOS', 'XPOS', 'FEATS', 'HEAD', 'DEPREL', 'DEPS', 'MISC']

# Create a list to collect processed rows
processed_rows = []

for i, row in df.iterrows():
    if isinstance(row[0], str) and row[0].strip().startswith("# text"):
        processed_rows.append([row[0]] + [""] * (len(columns)-1))  # keep the comment line
        continue
    
    if pd.isna(row[0]):
        processed_rows.append([""] * len(columns))  # empty line
        continue
    
    # Fill missing columns with "_"
    row = row.fillna("_").tolist()

    # Only process rows with enough fields
    if len(row) < 10:
        row += ["_"] * (10 - len(row))

    # Apply transformation if FEATS is not "_"
    if row[5] != "_":
        row[9] = "_"  # MISC
        row[8] = row[7]  # DEPREL = HEAD
        row[7] = row[6]  # HEAD = FEATS
        row[6] = row[5]  # FEATS = original FEATS
        row[5] = "_"     # FEATS = "_"
    
    # Ensure MISC is "_"
    if len(row) < 10 or row[9] == "":
        row[9] = "_"
    
    processed_rows.append(row)

# Create DataFrame with column names
final_df = pd.DataFrame(processed_rows, columns=columns)

# Save to Excel
final_df.to_excel(output_file, index=False)

print(f"Processed file saved to {output_file}")
