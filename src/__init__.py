from transformers import BertModel
from losses import CustomMSELoss, CustomJaccardLoss
from models import SimilarityModel_ReLU
from utils import train_model
from constants import DATASET_PATH
import torch

torch.manual_seed(42)
torch.use_deterministic_algorithms(True)

if torch.cuda.is_available():
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')

# Cargar el conjunto de datos desde tu archivo CSV

# BERT model pre-trained
model = BertModel.from_pretrained("bert-base-uncased")

# Instance of the similarity model
similarity_model_relu = SimilarityModel_ReLU(model, device)
# similarity_model_none = SimilarityModel_None(model)
# similarity_model_cosine = SimilarityModel_Cosine(model)

# Loss instance
rmse_criterion = CustomMSELoss(device)

# Model training
train_model(DATASET_PATH, similarity_model_relu, rmse_criterion, save_path="RELU_RMSE.png")
# train_model(DATASET_PATH, similarity_model_none, rmse_criterion, save_path="NONE_RMSE.png")
# train_model(DATASET_PATH, similarity_model_cosine, rmse_criterion, save_path="COSINE_RMSE.png")

## Same with jaccard loss
jaccard_loss = CustomJaccardLoss(device)

train_model(DATASET_PATH, similarity_model_relu, jaccard_loss, save_path="RELU_JACCARD.png")
# train_model(DATASET_PATH, similarity_model_none, jaccard_loss, save_path="NONE_JACCARD.png")
# train_model(DATASET_PATH, similarity_model_cosine, jaccard_loss, save_path="COSINE_JACCARD.png")

