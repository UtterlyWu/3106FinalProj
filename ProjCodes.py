from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader

# Load pre-trained SBERT model
model = SentenceTransformer("sentence-transformers/paraphrase-MiniLM-L6-v2")

# Prepare dataset
train_examples = [
    InputExample(texts=["Sentence1", "Sentence2"], label=1),  # Agree
    InputExample(texts=["Sentence3", "Sentence4"], label=0)   # Disagree
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