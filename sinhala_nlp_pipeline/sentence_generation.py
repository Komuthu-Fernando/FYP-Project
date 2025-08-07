import random

# Define word pools
names = ['තිස්ස', 'තිශ', 'සුමන ගුප්ත', 'ගාමිණී ධම්මරාජ', 'සෝනගුප්ත', 'සමනක', 'සේන', 'සමුද්ද', 'සුමන', 'මිත්තදේව', 'මුලගුප්ත', 'උත්තිය', 'නන්දක', 'වේලු', 'කනිසත', 'නාග', 'ගුටක තිස්ස', 'අභිජි ගුත්ත', 'ශිව', 'ආනන්ද', 'නන්දික', 'මානාපදස්සන', 'උපාලි', 'දානාටක', 'තිස්සා']
titles = ['ප්‍රධානියා', 'ගම්පාලක', 'භාණ්ඩාගාරික', 'මහා රජු', 'ප්‍රධානියාග', 'ගෘහපති', 'මැණික් කපන්නා', 'මාර්ග අධ්යක්ෂක', 'මැණික් කපන්නා']
roles = ['පුත්', 'දියණිය', 'පුත්‍රයා']
objects = ['ලෙන', 'ගුහාව', 'පියගැට පෙළ']
actions = ['පූජා', 'දෙනු', 'පිරිනමනු', 'පූජ']
recipients = ['සංඝයා']

# Determine verb ending based on action
def get_ending(action):
    if action == 'පිරිනමනු':
        return 'ලදි'
    elif action == 'දෙනු':
        return 'ලැබේ'
    elif action in ['පූජා', 'පූජ']:
        return 'කරන ලදි'
    return 'ලදි'

# Generate sentence pairs with varied structures
def generate_pair():
    # Randomly choose one of the three structures
    structure = random.randint(1, 3)
    
    # Select random elements
    name = random.choice(names)
    title = random.choice(titles)
    role = random.choice(roles)
    obj = random.choice(objects)
    action = random.choice(actions)
    recipient = random.choice(recipients)
    ending = get_ending(action)
    
    # Structure 1
    if structure == 1:
        incorrect = f"{title} {name} {role} {obj} {recipient} {action}"
        # 20% chance to use alternative correct form
        if random.random() < 0.2:
            correct = f"{title} {name}ගේ {role}ගේ {obj} {recipient}ට දෙන ලදී"
        else:
            correct = f"{title} {name}ගේ {role}ගේ {obj} {recipient}ට {action} {ending}"
    
    # Structure 2
    elif structure == 2:
        incorrect = f"{name} {obj} {recipient}"
        correct = f"{name}ගේ {obj} {recipient}ට {action} {ending}"
    
    # Structure 3
    else:
        incorrect = f"{name} {title} {role} {obj} {recipient} {action}"
        correct = f"{name} නැමති {title}ගේ {obj} {recipient}ට {action} {ending}"
    
    return f"input: {incorrect} → output: {correct}"

# Write sentence pairs to file
def write_dataset(filename="sinhala_pairs.txt", count=1900):
    with open(filename, "w", encoding="utf-8") as f:
        for _ in range(count):
            line = generate_pair()
            f.write(line + "\n")

# Generate the dataset
write_dataset()
print("Dataset generated as sinhala_pairs.txt")