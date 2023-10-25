import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, AdamW, get_linear_schedule_with_warmup

from constants import MAX_LENGTH, BATCH_SIZE


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


def data_process(path: str):
    """
    Process data for a natural language processing task using BERT.

    Args:
        path (str): Path to the CSV file containing the data with columns 'abstract', 'paraphrase_abstract', and 'Score (0-10)'.

    Returns:
        Tuple[DataLoader, DataLoader, pd.DataFrame, pd.DataFrame]: A tuple containing the following elements:
        1. train_loader (DataLoader): DataLoader for the training dataset.
        2. test_loader (DataLoader): DataLoader for the testing dataset.
        3. train_df (pd.DataFrame): DataFrame for the training data.
        4. test_df (pd.DataFrame): DataFrame for the testing data.
    
    This function reads the data from a CSV file, splits it into training and testing sets, tokenizes the text using BERT tokenizer,
    and returns DataLoaders for both sets along with the corresponding DataFrames.

    Example usage:
    ```
    train_loader, test_loader, train_df, test_df = data_process("data.csv")
    ```
    """
    
    df = pd.read_csv(path, usecols=["abstract", "paraphrase_abstract", "Score (0-10)"])

    # Divide tus datos en entrenamiento y prueba
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

    # Tokenizador BERT
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    train_dataset = CustomDataset(train_df, tokenizer, MAX_LENGTH)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    test_dataset = CustomDataset(test_df, tokenizer, MAX_LENGTH)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    return train_loader, test_loader, train_df, test_df


def train_model(train_loader: DataLoader, model: torch.nn.Module, criterion: torch.nn.Module, num_epochs: int = 5, learning_rate: int = 1e-5):
    """
    Train a PyTorch model using the specified data loader, loss criterion, and hyperparameters.

    Args:
        train_loader (DataLoader): DataLoader containing training data.
        model (torch.nn.Module): The PyTorch model to be trained.
        criterion (torch.nn.Module): Loss criterion for training.
        num_epochs (int, optional): Number of training epochs (default is 5).
        learning_rate (float, optional): Learning rate for the optimizer (default is 1e-5).

    Returns:
        None

    This function performs training for a specified number of epochs. It uses the AdamW optimizer with a learning rate
    scheduler to update the model's parameters. Training progress is printed at each epoch, displaying the average loss.

    Example usage:
    train_model(train_loader, model, criterion, num_epochs=10, learning_rate=1e-4)
    """
    
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=len(train_loader) * num_epochs)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0

        for batch in train_loader:
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            score = batch["score"]

            optimizer.zero_grad()
            output = model(input_ids, attention_mask)
            loss = criterion(output, score.view(-1, 1))
            loss.backward()
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()

        average_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {average_loss}")


def test_model(test_loader: DataLoader, test_df: pd.DataFrame, model: torch.nn.Module):
    """
    Test a PyTorch neural network model's performance on a given test dataset.

    Parameters:
    - test_loader (DataLoader): A PyTorch DataLoader containing the test dataset.
    - test_df (pd.DataFrame): A Pandas DataFrame containing the test data, including a "Score (0-10)" column.
    - model (torch.nn.Module): A PyTorch neural network model for prediction.

    Returns:
    - correlation (float): The Pearson correlation coefficient between the model's predictions and the true scores.

    The function evaluates the model's performance on the test data by making predictions and calculating
    the Pearson correlation coefficient between the predicted scores and the true scores in the test DataFrame.

    Example:
    >>> test_loader = DataLoader(test_dataset, batch_size=32)
    >>> correlation = test_model(test_loader, test_data, my_model)
    >>> print(f"Correlation with test data: {correlation}")
    """

    model.eval()
    predictions = []

    for batch in test_loader:
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        with torch.no_grad():
            output = model(input_ids, attention_mask)
        predictions.extend(output)

    # Calcula la correlación entre las predicciones y el campo "Score" en el conjunto de prueba
    predicted_scores = [p.item() for p in predictions]
    true_scores = test_df["Score (0-10)"].tolist()

    correlation = np.corrcoef(predicted_scores, true_scores)[0, 1]
    return correlation
