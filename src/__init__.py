from transformers import BertModel
from losses import CustomMSELoss, JaccardSimilarity
from models import SimilarityModel_ReLU, SimilarityModel_Cosine, SimilarityModel_None
from utils import train_model
from constants import DATASET_PATH
import torch
import torch.nn as nn

torch.manual_seed(42)
torch.use_deterministic_algorithms(True)


if torch.cuda.is_available():
    device = torch.device('cuda')
    num_devices = torch.cuda.device_count()
else:
    device = torch.device('cpu')
    num_devices = 1

# Cargar el conjunto de datos desde tu archivo CSV

# BERT model pre-trained
model = BertModel.from_pretrained("bert-base-uncased")

# Instance of the similarity model
similarity_model_relu = SimilarityModel_ReLU(model, device)
similarity_model_none = SimilarityModel_None(model, device)
similarity_model_cosine = SimilarityModel_Cosine(model, device)

if num_devices > 1:
    model = nn.DataParallel(similarity_model_relu)

# Loss instance
rmse_criterion = CustomMSELoss(device)

# Model training
train_model(DATASET_PATH, similarity_model_relu, rmse_criterion, save_path="RELU_RMSE.png")
# train_model(DATASET_PATH, similarity_model_none, rmse_criterion, save_path="NONE_RMSE.png")
train_model(DATASET_PATH, similarity_model_cosine, rmse_criterion, save_path="COSINE_RMSE.png")

## Same with jaccard loss
jaccard_loss = JaccardSimilarity(device)

train_model(DATASET_PATH, similarity_model_relu, jaccard_loss, save_path="RELU_JACCARD.png")
# train_model(DATASET_PATH, similarity_model_none, jaccard_loss, save_path="NONE_JACCARD.png")
train_model(DATASET_PATH, similarity_model_cosine, jaccard_loss, save_path="COSINE_JACCARD.png")

