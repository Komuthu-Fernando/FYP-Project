def split_conllu(input_file, train_file, dev_file, test_file, train_ratio=0.8, dev_ratio=0.1):
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    sentences = []
    current_sent = []
    for line in lines:
        if line.strip() == '' and current_sent:
            sentences.append(current_sent)
            current_sent = []
        else:
            current_sent.append(line)
    if current_sent:
        sentences.append(current_sent)
    
    total = len(sentences)
    train_end = int(total * train_ratio)
    dev_end = train_end + int(total * dev_ratio)
    
    with open(train_file, 'w', encoding='utf-8') as f:
        for sent in sentences[:train_end]:
            f.writelines(sent)
            f.write('\n')
    with open(dev_file, 'w', encoding='utf-8') as f:
        for sent in sentences[train_end:dev_end]:
            f.writelines(sent)
            f.write('\n')
    with open(test_file, 'w', encoding='utf-8') as f:
        for sent in sentences[dev_end:]:
            f.writelines(sent)
            f.write('\n')

split_conllu('data/depparse/si_stb-ud-test.conllu', 
             'data/depparse/si_stb-ud-train.conllu', 
             'data/depparse/si_stb-ud-dev.conllu', 
             'data/depparse/si_stb-ud-test.conllu')