import pandas as pd

# Load conllu data and tagged word data
conllu_df = pd.read_excel("updated_conllu_data.xlsx")
tagged_df = pd.read_excel("tagged_words.xlsx")

# Merge UPOS and XPOS where available
def fill_tags(row):
    match = tagged_df[tagged_df['FORM'] == row['FORM']]
    if not match.empty:
        if pd.isna(row['UPOS']) or row['UPOS'] == "":
            row['UPOS'] = match.iloc[0]['UPOS'] if pd.notna(match.iloc[0]['UPOS']) else row['UPOS']
        if pd.isna(row['XPOS']) or row['XPOS'] == "":
            row['XPOS'] = match.iloc[0]['XPOS'] if pd.notna(match.iloc[0]['XPOS']) else row['XPOS']
    return row

# Apply the tagging function
conllu_df = conllu_df.apply(fill_tags, axis=1)

# Save the final file
conllu_df.to_excel("final_tagged_conllu.xlsx", index=False)
print("final_tagged_conllu.xlsx created.")
