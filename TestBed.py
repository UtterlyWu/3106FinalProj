import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
import transformers
from sentence_transformers import SentenceTransformer, InputExample
from SBERTAgreementModel import SBERTAgreementModel
from tqdm import tqdm

# Load the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SBERTAgreementModel(num_classes=2)
model.load_state_dict(torch.load('intakemodels/intakeM_E5_L0.352_agree_v3.pth', weights_only=True)) 
model.to(device)  # Ensure the model is on the right device (GPU or CPU)
model.eval()

df = pd.read_csv("Dataset/labeled_data.csv", usecols=['label','body_parent','body_child'])
test_df = df.iloc[41000:42895]#13500]
df.loc[df['label'] != 2, 'label'] = 0
df.loc[df['label'] == 2, 'label'] = 1

# df.loc[df['label'] != 0, 'label'] = -1
# df.loc[df['label'] == 0, 'label'] = 1
# df.loc[df['label'] == -1, 'label'] = 0

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
# raw_predictions = []
correct_predictions = 0
with torch.no_grad():
    for sentence_pairs, labels in test_dataloader: #tqdm(test_dataloader):
        #The progress bar says out of 60 because it's 60 batches, not examples. Each batch is 32 examples.
        sentence1, sentence2 = zip(*sentence_pairs)

        outputs = model(list(sentence1), list(sentence2))
        # outputs = model(["I disagree with that"], ["I agree"])
        predicted_classes = torch.argmax(outputs, dim=1)

        # print()
        # if (predicted_classes == 1):
        #     print("The response agreed")
        # else:
        #     print("The response did not agree")
        # exit()

        correct_predictions += (predicted_classes == torch.tensor(labels).to(device)).sum().item()
        # for output in zip(outputs.tolist()):
        #     raw_predictions.append([output])
print()
# print(raw_predictions)
print(f"Test accuracy was: {(correct_predictions/total)*100}%")