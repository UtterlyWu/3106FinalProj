import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
import transformers
from sentence_transformers import SentenceTransformer, InputExample, losses
from SBERTAgreementModel import SBERTAgreementModel

# Load the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SBERTAgreementModel(base_model_name='sentence-transformers/stsb-distilroberta-base-v2', num_classes=3)  # Same as when you trained
model.load_state_dict(torch.load('sbert_agreement_model.pth'))  # Load saved weights
model.to(device)  # Ensure the model is on the right device (GPU or CPU)
model.eval()

# Input sentences

sentences = [
    ["Joe biden is an idiot. He's practically geriatric.", "What are you talking about? Do you know the number of things he's done for this country? You're probably racist."],
    ["Joe biden is an idiot. He's practically geriatric.", "Dude is too old to be running the country"],
    ["This sucks", "Yeah it does"],
    ["He's gonna win,", "Bullshit"],
    ["Are we back?", "Nah it's over"],
    ["We're so back?", "WE'RE SO FUCKING BAAACK!"],
    ["The sky is blue", "The sky has a blue hue"],
]

df = pd.read_csv("Debagreement/labeled_data.csv", usecols=['label','body_parent','body_child'])
test_df = df.iloc[39000:42000]#13500]

def custom_collate_fn(batch):
    """
    Custom function to collate batches for the DataLoader.
    """
    sentence_pairs = [item[0] for item in batch]  # Extract the sentence pairs
    labels = [item[1] for item in batch]  # Extract the labels
    return sentence_pairs, labels

# Define a dataset loader
def create_dataloader(examples, batch_size):
    texts = [(ex.texts[0], ex.texts[1]) for ex in examples]
    labels = [ex.label for ex in examples]
    dataset = list(zip(texts, labels))
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate_fn)

test_examples = [
    InputExample(texts=[row['body_parent'], row['body_child']], label=row['label'])
    for _, row in test_df.iterrows()
]
batch_size = 32

test_dataloader = create_dataloader(test_examples, batch_size)

total = len(test_examples)
correct_predictions = 0
with torch.no_grad():
    for sentence_pairs, labels in test_dataloader:
        sentence1, sentence2 = zip(*sentence_pairs)
        outputs = model(list(sentence1), list(sentence2))
        predicted_classes = torch.argmax(outputs, dim=1)
        correct_predictions += (predicted_classes == torch.tensor(labels).to(device)).sum().item()

        # for s1, s2, label, prediction in zip(sentence1, sentence2, labels, predicted_classes):
        #     print(f"Sentence 1: {s1}")
        #     print(f"Sentence 2: {s2}")
        #     print(f"True Label: {label}")
        #     print(f"Predicted Label: {prediction}")
        #     print("-" * 50)  # Separator for readability
print()
print(f"Test accuracy was: {(correct_predictions/total)*100}%")

print()
# Forward pass
# for sentencepair in sentences:
#     output = model([sentencepair[0]], [sentencepair[1]])
#     predicted_class = torch.argmax(output, dim=1).item()
#     print()
#     print(f"'{sentencepair[0]}'")
#     print(f"'{sentencepair[1]}'")

#     readable_class = {0 : "Sentences are disagreeing.", 1 : "Unsure if the sentences agree", 2 : "Sentences are agreeing"}
#     print(f"Predicted Class: {readable_class[predicted_class]}")