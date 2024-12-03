import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
import transformers
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer, InputExample, losses
# from google.colab import drive 
# from google.colab import files
import io

class SBERTAgreementModel(nn.Module):
    def __init__(self, num_classes):
        super(SBERTAgreementModel, self).__init__()
        # establish device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self.agree = SentenceTransformer('transformers/agree_ft_mpnet_Constrast/checkpoint-14000')
        self.agree = SentenceTransformer('transformers/checkpoint-14000')
        # self.disagree = SentenceTransformer('transformers/disagree_ft_mpnet_Constrast/checkpoint-12000')
        self.disagree = SentenceTransformer('transformers/checkpoint-12000')
        # Add a classification head
        self.classifier = nn.Sequential(
            nn.Linear(768 * 2 + 768 * 2, 512),  # 768: embedding size from SBERT
            nn.ReLU(),
            nn.Dropout(0.6),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, sentence1, sentence2):
        # Encode sentence pairs with sentence transformer
        #sem_diff = torch.abs(self.semantic.encode(sentence1, convert_to_tensor=True) - self.semantic.encode(sentence2, convert_to_tensor=True))
        agree_em1 = self.agree.encode(sentence1, convert_to_tensor=True)
        disagree_em1 = self.disagree.encode(sentence1, convert_to_tensor=True)
        agree_em2 = self.agree.encode(sentence2, convert_to_tensor=True)
        disagree_em2 = self.disagree.encode(sentence2, convert_to_tensor=True)
        # nli_emb2 = self.nli.encode(sentence2, convert_to_tensor=True)
        combined_embeddings = torch.cat([agree_em1, disagree_em1, agree_em2, disagree_em2], dim=1)

        # Pass through the classification head
        output = self.classifier(combined_embeddings)
        return output
    
    # def forward(self, sentence1, sentence2):
    #     combined = [f"{s1} !|! {s2}" for s1, s2 in zip(sentence1, sentence2)]
    #     nli = self.nli.encode(combined, convert_to_tensor=True)
    #     nli = nli.to(self.device)
    #     # Pass through the classification head
    #     output = self.classifier(nli)
    #     return output