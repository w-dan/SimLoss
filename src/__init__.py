from transformers import BertModel

# Cargar el conjunto de datos desde tu archivo CSV

from losses import CustomMSELoss, CustomJaccardLoss
from models import SimilarityModel_ReLU, SimilarityModel_None, SimilarityModel_Cosine
from utils import data_process, test_model, train_model
from constants import DATASET_PATH, NUM_EPOCHS


# Process data
train_loader, test_loader, train_df, test_df = data_process(DATASET_PATH)

# BERT model pre-trained
model = BertModel.from_pretrained("bert-base-uncased")

# Instance of the similarity model
similarity_model_relu = SimilarityModel_ReLU(model)
similarity_model_none = SimilarityModel_None(model)
similarity_model_cosine = SimilarityModel_Cosine(model)

# Loss instance
rmse_loss = CustomMSELoss()

# Model training
train_model(train_loader, similarity_model_relu, rmse_loss)
train_model(train_loader, similarity_model_none, rmse_loss)
train_model(train_loader, similarity_model_cosine, rmse_loss)

# Model testing
relu_correlation = test_model(test_loader, test_df, similarity_model_relu)
none_correlation = test_model(test_loader, test_df, similarity_model_none)
cosine_correlation = test_model(test_loader, test_df, similarity_model_cosine)

print(f"Correlación entre las predicciones y el campo 'Score' con ReLU: {relu_correlation}")
print(f"Correlación entre las predicciones y el campo 'Score' con None: {none_correlation}")
print(f"Correlación entre las predicciones y el campo 'Score' con Cosine: {cosine_correlation}")

## Same with jaccard loss
jaccard_loss = CustomJaccardLoss()

# ....
# train_model(train_loader, similarity_model_relu, jaccard_loss)
# train_model(train_loader, similarity_model_none, jaccard_loss)
# train_model(train_loader, similarity_model_cosine, jaccard_loss)
