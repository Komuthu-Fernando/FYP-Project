input_path = "input.conllu"    # your original file
output_path = "cleaned.conllu" # new output file

def clean_file(input_path, output_path):
    with open(input_path, 'r', encoding='utf-8') as infile:
        lines = infile.readlines()

    cleaned_lines = []
    temp_block = []

    for line in lines:
        stripped = line.strip()

        if stripped.startswith("# text"):
            # If starting new block, flush previous one if any
            if temp_block:
                cleaned_lines.extend(temp_block)
                cleaned_lines.append('')  # add clean blank line
                temp_block = []
            temp_block.append(stripped)
        elif stripped == "":
            continue  # skip whitespace-only lines
        else:
            temp_block.append(stripped)

    # Flush the final block
    if temp_block:
        cleaned_lines.extend(temp_block)
        cleaned_lines.append('')

    # Write to output
    with open(output_path, 'w', encoding='utf-8') as outfile:
        outfile.write('\n'.join(cleaned_lines))

    print(f"Cleaned file saved to {output_path}")

# Run the function
clean_file(input_path, output_path)
