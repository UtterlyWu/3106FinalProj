import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, InputExample, losses
from SBERTAgreementModel import SBERTAgreementModel
import time

class Comparer:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        self.model = SBERTAgreementModel(num_classes=2)

        model_path = 'intakeM_E5_L0.352_agree_v3.pth'
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

    def compare(self, sentence1, sentence2):
        with torch.no_grad():
            outputs = self.model([sentence1], [sentence2])
            predicted_class = torch.argmax(outputs, dim=1)
            return predicted_class.item()  # Convert tensor to Python integer

c = Comparer()
result = c.compare("This guy sucks", "No man you suck")
print(f"Predicted class: {result}")
