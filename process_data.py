# This implementation is adapted from ReCall: https://github.com/ruoyuxie/recall
import random
from datasets import load_dataset
import json
import nltk
from nltk.corpus import wordnet

# Download required NLTK data
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('averaged_perceptron_tagger_eng')

def save_jsonl(data, path):
    with open(path, "w", encoding="utf-8") as f:
        for example in data:
            f.write(json.dumps(example, ensure_ascii=False) + "\n")

def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return None

def delete_portion_of_words(text, portion):
    words = text.split()
    num_words_to_delete = int(len(words) * portion)
    words_to_delete = random.sample(range(len(words)), num_words_to_delete)
    new_words = [word for i, word in enumerate(words) if i not in words_to_delete]
    return ' '.join(new_words)

def substitute_synonyms(text, portion):
    words = text.split()
    num_words_to_substitute = int(len(words) * portion)
    words_to_substitute = random.sample(range(len(words)), num_words_to_substitute)
    
    tagged_words = nltk.pos_tag(words)
    
    for index in words_to_substitute:
        word, pos = tagged_words[index]
        wordnet_pos = get_wordnet_pos(pos)
        
        if wordnet_pos:
            synsets = wordnet.synsets(word, pos=wordnet_pos)
            if synsets:
                synonyms = [lemma.name() for synset in synsets for lemma in synset.lemmas() if lemma.name() != word]
                if synonyms:
                    words[index] = random.choice(synonyms).replace('_', ' ')
    return ' '.join(words)

def apply_attack(text, attack_type, attack_strength):
    if attack_type == "del":
        return delete_portion_of_words(text, attack_strength)
    elif attack_type == "sub":
        return substitute_synonyms(text, attack_strength)
    else:
        return text  # No attack or 'para' (paraphrased data is handled separately)

def create_dataset(dataset_name, sub_dataset_name, output_dir, num_shots, attack_type=None, attack_strength=0):
    if attack_type == 'para':
        dataset_name = "WikiMIA_paraphrased_perturbed"
    if dataset_name == "wikimia":
        sub_dataset_name = int(sub_dataset_name)
        dataset = load_dataset("swj0419/WikiMIA", split=f"WikiMIA_length{sub_dataset_name}", token=True)
    elif dataset_name == "WikiMIA_paraphrased_perturbed":
        sub_dataset_name = int(sub_dataset_name)
        dataset = load_dataset("zjysteven/WikiMIA_paraphrased_perturbed", split=f"WikiMIA_length{sub_dataset_name}_paraphrased", token=True)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}. Please modify the code to include the dataset.")

    member_data = []
    nonmember_data = []

    if dataset_name in ["wikimia", "WikiMIA_paraphrased_perturbed"]:
        for data in dataset:
            text = apply_attack(data["input"], attack_type, attack_strength)
            if data["label"] == 1:
                member_data.append(text)
            elif data["label"] == 0:
                nonmember_data.append(text)
    else:
        member_data = [apply_attack(text, attack_type, attack_strength) for text in dataset['member']]
        nonmember_data = [apply_attack(text, attack_type, attack_strength) for text in dataset['nonmember']]

    random.shuffle(member_data)
    random.shuffle(nonmember_data)
    num_shots = int(num_shots)

    nonmember_prefix = nonmember_data[:num_shots]
    nonmember_data = nonmember_data[num_shots:]
    member_data_prefix = member_data[:num_shots]
    member_data = member_data[num_shots:]

    full_data = []
    for nm_data, m_data in zip(nonmember_data, member_data):
        full_data.append({"input": nm_data, "label": 0})
        full_data.append({"input": m_data, "label": 1})
    
    return full_data, nonmember_prefix, member_data_prefix