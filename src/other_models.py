import torch
from torch.utils.data import DataLoader, Dataset
from torchmetrics import JaccardIndex
from transformers import BertTokenizer, BertModel, AdamW, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
import numpy as np

# Define una capa de regresión lineal con activación cosine
class SimilarityModel_Cosine(torch.nn.Module):
    def __init__(self, model):
        super(SimilarityModel_Cosine, self).__init__()
        self.bert = model
        self.fc1 = torch.nn.Linear(768, 128)  # 768 es la dimensión de salida de BERT
        self.cosine = torch.nn.CosineSimilarity()
        self.fc2 = torch.nn.Linear(128, 1)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids, attention_mask)
        pooled_output = outputs["last_hidden_state"][:, 0]
        x = self.fc1(pooled_output)
        x = self.cosine(x)
        score = self.fc2(x)
        return score

# Función de pérdida y optimizador
# Función de pérdida personalizada para manejar valores "nan"
class CustomJaccardLoss(torch.nn.Module):
    def __init__(self):
        super(CustomJaccardLoss, self).__init__()

    def forward(self, output, target):
        # Encuentra los índices donde hay valores "nan"
        nan_indices = torch.isnan(output) | torch.isnan(target)
        
        # Calcula la pérdida MSELoss solo para valores no "nan"
        loss = JaccardIndex(output[~nan_indices], 1 - target[~nan_indices])

        return loss
