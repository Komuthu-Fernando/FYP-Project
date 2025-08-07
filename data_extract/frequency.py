import pandas as pd

# Load the updated CONLLU data
df = pd.read_excel("updated_conllu_data.xlsx")

# Drop rows with empty FORM values
df = df[df['FORM'].notna() & (df['FORM'].astype(str).str.strip() != "")]

# Count frequencies
word_freq = df['FORM'].value_counts().reset_index()
word_freq.columns = ['FORM', 'FREQUENCY']

# Export to Excel
word_freq.to_excel("word_frequencies.xlsx", index=False)
print("word_frequencies.xlsx created.")
