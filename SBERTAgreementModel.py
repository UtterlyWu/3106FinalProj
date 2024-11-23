import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
import transformers
from sentence_transformers import SentenceTransformer, InputExample, losses
from google.colab import drive 
from google.colab import files
import io

class SBERTAgreementModel(nn.Module):
    def __init__(self, base_model_name, num_classes):
        super(SBERTAgreementModel, self).__init__()
        # Load pre-trained SBERT model
        self.sbert = SentenceTransformer(base_model_name)
        # Add a classification head
        self.classifier = nn.Sequential(
            nn.Linear(768 * 3, 512),  # 768: embedding size from SBERT
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes),
        )

    def forward(self, sentence1, sentence2):
        # Encode sentence pairs with SBERT
        embeddings1 = self.sbert.encode(sentence1, convert_to_tensor=True)
        embeddings2 = self.sbert.encode(sentence2, convert_to_tensor=True)

        # Combine embeddings (e.g., element-wise difference and concatenation)
        diff_em = torch.abs(embeddings1 - embeddings2)
        # avg_em = (embeddings1 + embeddings2) / 2
        combined_embeddings = torch.cat([embeddings1, embeddings2, diff_em], dim=1)
        # embedding = self.sbert.encode([sentence1, sentence2], convert_to_tensor=True)

        # Pass through the classification head
        output = self.classifier(combined_embeddings)
        return output