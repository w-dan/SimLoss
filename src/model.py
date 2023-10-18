import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertModel, AdamW, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
import numpy as np

# Cargar el conjunto de datos desde tu archivo CSV
import pandas as pd

# Supongamos que tienes un archivo CSV llamado "dataset.csv" con tres columnas: "abstract", "paraphrase_abstract", "Score"
df = pd.read_csv("../data/SimLoss.csv", usecols=["abstract", "paraphrase_abstract", "Score (0-10)"])

# Divide tus datos en entrenamiento y prueba
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# Tokenizador BERT
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Clase personalizada para el conjunto de datos
class CustomDataset(Dataset):
    def __init__(self, data, tokenizer, max_length):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        abstract = str(self.data.iloc[index]["abstract"])
        paraphrase = str(self.data.iloc[index]["paraphrase_abstract"])
        score = float(self.data.iloc[index]["Score (0-10)"])
        score = torch.tensor(score, dtype=torch.float32)
        
        inputs = self.tokenizer(abstract, paraphrase, return_tensors="pt", max_length=self.max_length, padding="max_length", truncation=True)
        
        return {
            "input_ids": inputs["input_ids"].flatten(),
            "attention_mask": inputs["attention_mask"].flatten(),
            "score": score
        }

# Configuración del modelo y del DataLoader
max_length = 128
batch_size = 32

train_dataset = CustomDataset(train_df, tokenizer, max_length)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Modelo BERT pre-entrenado
model = BertModel.from_pretrained("bert-base-uncased")

# Define una capa de regresión lineal con activación ReLU
class SimilarityModel(torch.nn.Module):
    def __init__(self, model):
        super(SimilarityModel, self).__init__()
        self.bert = model
        self.fc1 = torch.nn.Linear(768, 128)  # 768 es la dimensión de salida de BERT
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(128, 1)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids, attention_mask)
        pooled_output = outputs["last_hidden_state"][:, 0]
        x = self.fc1(pooled_output)
        x = self.relu(x)
        score = self.fc2(x)
        return score

# Instancia del modelo de similitud
similarity_model = SimilarityModel(model)

# Entrenamiento del modelo
num_epochs = 5

# Función de pérdida y optimizador
# Función de pérdida personalizada para manejar valores "nan"
class CustomMSELoss(torch.nn.Module):
    def __init__(self):
        super(CustomMSELoss, self).__init__()

    def forward(self, output, target):
        # Encuentra los índices donde hay valores "nan"
        nan_indices = torch.isnan(output) | torch.isnan(target)
        
        # Calcula la pérdida MSELoss solo para valores no "nan"
        loss = torch.mean((output[~nan_indices] - target[~nan_indices]) ** 2)

        return loss
criterion = CustomMSELoss()
optimizer = AdamW(similarity_model.parameters(), lr=1e-5)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=len(train_loader) * num_epochs)

for epoch in range(num_epochs):
    similarity_model.train()
    total_loss = 0.0

    for batch in train_loader:
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        score = batch["score"]

        optimizer.zero_grad()
        output = similarity_model(input_ids, attention_mask)
        loss = criterion(output, score.view(-1, 1))
        loss.backward()
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()

    average_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {average_loss}")

# Evaluación del modelo (Puedes usar un conjunto de prueba similar al de entrenamiento)

# Usar el modelo para predecir similitud entre abstract y paraphrase_abstract en tu conjunto de prueba
test_dataset = CustomDataset(test_df, tokenizer, max_length)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

similarity_model.eval()
predictions = []

for batch in test_loader:
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    with torch.no_grad():
        output = similarity_model(input_ids, attention_mask)
    predictions.extend(output)

# Calcula la correlación entre las predicciones y el campo "Score" en el conjunto de prueba
predicted_scores = [p.item() for p in predictions]
true_scores = test_df["Score (0-10)"].tolist()

correlation = np.corrcoef(predicted_scores, true_scores)[0, 1]
print(f"Correlación entre las predicciones y el campo 'Score': {correlation}")
