from collections import defaultdict
from typing import Counter
import pandas as pd
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader


def import_datasets(path,threshold):
    df = pd.read_csv('Dataset\\labeled_data.csv')
    agreement_Label = df.iloc[:threshold, 0].to_numpy()
    body_Parent = df.iloc[1:threshold, 4].to_numpy()
    body_Child = df.iloc[1:threshold, 5].to_numpy()
    return agreement_Label,body_Parent,body_Child
    

    return train_examples

def word_count_analysis():
    agreement_Label,body_Parent,body_Child = import_datasets("",10000)
    body_Child_unique_words = [list(set(child.split())) for child in body_Child]
    
    # Initialize a dictionary to store counts of words for each label
    word_label_counts = defaultdict(lambda: {0: 0, 1: 0, 2: 0})

    # Populate word_label_counts based on the label for each entry
    for label, words in zip(agreement_Label, body_Child_unique_words):
        for word in words:
            word_label_counts[word][label] += 1

    # Sort words by total occurrences across all labels in descending order
    sorted_word_counts = sorted(word_label_counts.items(), 
                                key=lambda item: sum(item[1].values()), 
                                reverse=True)
    #for i in range(100):
        #print(sorted_word_counts[i])
    return sorted_word_counts
    
def calculate_ratios(sorted_word_counts):
    # Select the top 100 words
    top_100_words = sorted_word_counts[:100]

    # Calculate the "offness" of the word distribution for each word
    word_ratios = []
    for word, counts in top_100_words:
        count_0 = counts[0]
        count_1 = counts[1]
        count_2 = counts[2]
        total = count_0 + count_1 + count_2
        
        # If the total count for a word is 0, skip this word (it has no occurrences)
        if total == 0:
            continue
        
        # Calculate proportions for each category
        prop_0 = count_0 / total
        prop_1 = count_1 / total
        prop_2 = count_2 / total
        
        # Sort proportions to find the max and second max
        sorted_props = sorted([prop_0, prop_1, prop_2], reverse=True)
        
        # The "offness" is the difference between the max and second max proportions
        offness = sorted_props[0] - sorted_props[1]
        
        # Append word, its ratio (0s/2s), and offness value
        word_ratios.append((word, (count_0 / count_2) if count_2 != 0 else float('inf'), offness))

    # Sort words by "offness" (the more unbalanced, the higher the offness)
    sorted_by_offness = sorted(word_ratios, key=lambda x: x[2], reverse=True)

    # Display the top words with offness
    for word, ratio, offness in sorted_by_offness:
        print(f"Word: '{word}' - Ratio (0s/2s): {ratio:.2f}, Offness: {offness:.2f}")

    return sorted_by_offness


def sbert_train():
    # Load pre-trained SBERT model
    model = SentenceTransformer("sentence-transformers/paraphrase-MiniLM-L6-v2")

    # Prepare dataset
    agreement_Label,body_Parent,body_Child = import_datasets("",1000)
    label_map = {0: 0.0, 1: 0.5, 2: 1.0}
    labels = [label_map[label] for label in agreement_Label]
    train_examples = [
        InputExample(texts=[parent, child], label=label)
        for parent, child, label in zip(body_Parent, body_Child, labels)
    ]
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)

    # Loss function
    train_loss = losses.CosineSimilarityLoss(model)

    # Fine-tune model
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=3,
        warmup_steps=100
    )

    # Save the fine-tuned model
    model.save("agreement_detection_sbert")
    return model