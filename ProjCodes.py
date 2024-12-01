import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from sentence_transformers import InputExample
from SBERTAgreementModel import SBERTAgreementModel
from tqdm import tqdm

df = pd.read_csv("Dataset/labeled_data.csv", usecols=['label','body_parent','body_child'])
df.loc[df['label'] != 2, 'label'] = 0
df.loc[df['label'] == 2, 'label'] = 1

# df.loc[df['label'] !=0, 'label'] = -1
# df.loc[df['label'] == 0, 'label'] = 1
# df.loc[df['label'] == -1, 'label'] = 0

train_df = df.head(39000)
val_df = df.iloc[39000:41000]
# analyzer = SentimentIntensityAnalyzer()

train_examples = [
    InputExample(texts=[row['body_parent'], row['body_child']], label=row['label'])
    for _, row in train_df.iterrows()
]

validation_examples = [
    InputExample(texts=[row['body_parent'], row['body_child']], label=row['label'])
    for _, row in val_df.iterrows()
]

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

# Training parameters
batch_size = 32
num_classes = 2
epochs = 8
learning_rate = 5e-5

# Initialize the model, loss, and optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SBERTAgreementModel(num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# Prepare dataloader
train_dataloader = create_dataloader(train_examples, batch_size)
validation_dataloader = create_dataloader(validation_examples, batch_size)

# Training loop
model.train()
for epoch in range(epochs):
    total_loss = 0
    for sentences, labels in tqdm(train_dataloader):
        sentence1, sentence2 = zip(*sentences)
        labels = torch.tensor(labels).to(device)

        # Forward pass
        outputs = model(list(sentence1), list(sentence2))
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    
    model.eval()  # Set model to evaluation mode
    total_val_loss = 0
    with torch.no_grad(): 
        for sentence_pairs, labels in validation_dataloader:
            sentence1, sentence2 = zip(*sentence_pairs)
            labels = torch.tensor(labels).to(device)

            # Forward pass
            outputs = model(list(sentence1), list(sentence2))
            loss = criterion(outputs, labels)
            total_val_loss += loss.item()
    
    torch.save(model.state_dict(), f"intakemodels/intakeM_E{epoch+1}_L{(total_val_loss / len(validation_dataloader)):.3f}_agree_v3.pth")

    print(f"Epoch {epoch+1}, Training Loss: {total_loss / len(train_dataloader)}, Validation Loss: {total_val_loss / len(validation_dataloader)}")