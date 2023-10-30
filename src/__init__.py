from transformers import BertModel
from losses import CustomMSELoss, JaccardSimilarity
from models import SimilarityModel_ReLU, SimilarityModel_Cosine, SimilarityModel_None
from utils import train_model
from constants import DATASET_PATH, OUTPUT_PATH
import torch
import torch.nn as nn
import os

torch.manual_seed(42)
torch.use_deterministic_algorithms(True)


if torch.cuda.is_available():
    device = torch.device('cuda')
    num_devices = torch.cuda.device_count()
else:
    device = torch.device('cpu')
    num_devices = 1

if not os.path.exists(OUTPUT_PATH):
    os.makedirs(OUTPUT_PATH, exist_ok=True)

# BERT model pre-trained
model = BertModel.from_pretrained("bert-base-uncased")

# Instance of the similarity model
similarity_model_relu = SimilarityModel_ReLU(model, device)
# similarity_model_none = SimilarityModel_None(model, device)
similarity_model_cosine = SimilarityModel_Cosine(model, device)

if num_devices > 1:
    model = nn.DataParallel(similarity_model_relu)

# Loss instance
rmse_criterion = CustomMSELoss(device)

# Model training
relu_mse_df = train_model(DATASET_PATH, similarity_model_relu, rmse_criterion, save_path=os.path.join(OUTPUT_PATH, "RELU_MSE.png"))
relu_mse_df.to_csv(os.path.join(OUTPUT_PATH, "RELU_MSE.csv"))
# train_model(DATASET_PATH, similarity_model_none, rmse_criterion, save_path="NONE_RMSE.png")
cosine_mse_df = train_model(DATASET_PATH, similarity_model_cosine, rmse_criterion, save_path=os.path.join(OUTPUT_PATH, "COSINE_MSE.png"))
cosine_mse_df.to_csv(os.path.join(OUTPUT_PATH, "COSINE_MSE.csv"))

## Same with jaccard loss
jaccard_loss = JaccardSimilarity(device)

relu_jaccard_df = train_model(DATASET_PATH, similarity_model_relu, jaccard_loss, save_path=os.path.join(OUTPUT_PATH, "RELU_JACCARD.png"))
relu_jaccard_df.to_csv(os.path.join(OUTPUT_PATH, "RELU_JACCARD.csv"))
# train_model(DATASET_PATH, similarity_model_none, jaccard_loss, save_path="NONE_JACCARD.png")
cosine_jaccard_df = train_model(DATASET_PATH, similarity_model_cosine, jaccard_loss, save_path=os.path.join(OUTPUT_PATH, "COSINE_JACCARD.png"))
cosine_jaccard_df.to_csv(os.path.join(OUTPUT_PATH, "COSINE_JACCARD.csv"))

