import torch
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, get_linear_schedule_with_warmup
from torch.optim import AdamW
from scipy.stats import pearsonr


from constants import MAX_LENGTH, BATCH_SIZE, NUM_EPOCHS


# Custom class for dataset
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

        inputs_abstract = self.tokenizer(
            abstract,
            return_tensors="pt",
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
        )
        inputs_paraphrase = self.tokenizer(
            paraphrase,
            return_tensors="pt",
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
        )

        return {
            "input_ids_abstract": inputs_abstract["input_ids"].flatten(),
            "attention_mask_abstract": inputs_abstract["attention_mask"].flatten(),
            "input_ids_paraphrase": inputs_paraphrase["input_ids"].flatten(),
            "attention_mask_paraphrase": inputs_paraphrase["attention_mask"].flatten(),
            "score": score,
        }


def data_process(path: str):
    """
    Process data for a natural language processing task using BERT.

    Args:
        path (str): Path to the CSV file containing the data with columns 'abstract', 'paraphrase_abstract', 'Score (0-10)', and 'Not/Bad generated'.

    Returns:
        Tuple[DataLoader, DataLoader, pd.DataFrame, pd.DataFrame]: A tuple containing the following elements:
        1. train_loader (DataLoader): DataLoader for the training dataset.
        2. test_loader (DataLoader): DataLoader for the testing dataset.
        3. val_loader (DataLoader): DataLoader for the validation dataset.
        4. train_df (pd.DataFrame): DataFrame for the training data.
        5. test_df (pd.DataFrame): DataFrame for the testing data.
        6. val_df (pd.DataFrame): DataFrame for the validation data.

    This function reads the data from a CSV file, filters the records with 'Not/Bad generated' as 'False', splits it into training, testing and validating sets, tokenizes the text using BERT tokenizer,
    and returns DataLoaders for all sets along with the corresponding DataFrames.

    Example usage:
    ```
    train_loader, test_loader, val_loader, train_df, test_df, val_df = data_process("data.csv")
    ```
    """

    df = pd.read_csv(path, usecols=["abstract", "paraphrase_abstract", "Score (0-10)", "Not/Bad generated"])
    df = df[df['Not/Bad generated'] == False]
    df.drop(columns=["Not/Bad generated"], inplace=True)

    # Dataset division in 'train' and 'test'
    train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

    # BERT Tokenizer
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    train_dataset = CustomDataset(train_df, tokenizer, MAX_LENGTH)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    test_dataset = CustomDataset(test_df, tokenizer, MAX_LENGTH)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    val_dataset = CustomDataset(val_df, tokenizer, MAX_LENGTH)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    return train_loader, test_loader, val_loader, train_df, test_df, val_df


def train_model(
    path: str,
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    learning_rate: float = 1e-5,
    save_path: str = "",
):
    """
    Train a PyTorch model using the specified data loader, loss criterion, and hyperparameters.

    Args:
        path (str): Path to the CSV file containing the data with columns 'abstract', 'paraphrase_abstract', and 'Score (0-10)'.
        model (torch.nn.Module): The PyTorch model to be trained.
        criterion (torch.nn.Module): Loss criterion for training.
        learning_rate (float, optional): Learning rate for the optimizer (default is 1e-5).
        save_path (str, optional): Path to save the plot of training progress (default is "").

    Returns:
        None

    This function performs training for a specified number of epochs. It uses the AdamW optimizer with a learning rate
    scheduler to update the model's parameters. Training progress is printed at each epoch, displaying the average loss.
    """

    train_loader, test_loader, val_loader, train_df, test_df, val_df = data_process(path)

    train_losses = []
    val_losses = []
    correlation_scores = []

    optimizer = AdamW(model.parameters(), lr=learning_rate)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=len(train_loader) * NUM_EPOCHS
    )
    
    for epoch in range(NUM_EPOCHS):
        model.train()
        total_loss = 0.0

        for batch in train_loader:
            input_ids_abstract = batch["input_ids_abstract"].to(criterion.device)
            attention_mask_abstract = batch["attention_mask_abstract"].to(criterion.device)
            input_ids_paraphrase = batch["input_ids_paraphrase"].to(criterion.device)
            attention_mask_paraphrase = batch["attention_mask_paraphrase"].to(criterion.device)
            score = batch["score"].to(criterion.device)

            optimizer.zero_grad()
            output = model(
                input_ids_abstract,
                attention_mask_abstract,
                input_ids_paraphrase,
                attention_mask_paraphrase,
            )

            loss = criterion(output, score.view(-1, 1))
            loss.backward()
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()

        average_loss = total_loss / len(train_loader)
        train_losses.append(average_loss)
        print(f"Epoch {epoch+1}/{NUM_EPOCHS}, Loss: {average_loss}")

        # Test the model
        correlation = test_model(test_loader, test_df, model)
        correlation_scores.append(correlation)

        # Validate the model
        average_val_loss = val_model(val_loader, model, criterion)
        val_losses.append(average_val_loss)

    print("LAST: ", len(train_losses))
    plot_model(correlation_scores, train_losses, val_losses, save_path)


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
    """

    model.eval()
    predictions = []

    for batch in test_loader:
        input_ids_abstract = batch["input_ids_abstract"]
        attention_mask_abstract = batch["attention_mask_abstract"]
        input_ids_paraphrase = batch["input_ids_paraphrase"]
        attention_mask_paraphrase = batch["attention_mask_paraphrase"]

        with torch.no_grad():
            output = model(
                input_ids_abstract,
                attention_mask_abstract,
                input_ids_paraphrase,
                attention_mask_paraphrase,
            )
        predictions.extend(output)

    # Calculate correlations between "Score" predictions in the test dataset
    predicted_scores = [p.item() for p in predictions]
    true_scores = test_df["Score (0-10)"].tolist()

    correlation, _ = pearsonr(predicted_scores, true_scores)  # Utilizar pearsonr
    print(
        f"Correlación entre las predicciones y el campo 'Score' en el conjunto de prueba: {correlation}"
    )
    return correlation

def val_model(
    val_loader: DataLoader, model: torch.nn.Module, criterion: torch.nn.Module
):
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch in val_loader:
            input_ids_abstract = batch["input_ids_abstract"]
            attention_mask_abstract = batch["attention_mask_abstract"]
            input_ids_paraphrase = batch["input_ids_paraphrase"]
            attention_mask_paraphrase = batch["attention_mask_paraphrase"]
            score = batch["score"]

            output = model(
                input_ids_abstract,
                attention_mask_abstract,
                input_ids_paraphrase,
                attention_mask_paraphrase,
            )
            loss = criterion(output, score.view(-1, 1))
            val_loss += loss.item()

    average_val_loss = val_loss / len(val_loader)
    return average_val_loss


def plot_model(
    correlation_scores: list, train_losses: list, val_losses: list, save_path: str = ""
):
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, NUM_EPOCHS + 1), train_losses, label="Entrenamiento")
    plt.plot(range(1, NUM_EPOCHS + 1), val_losses, label="Validación")
    plt.xlabel("Épocas")
    plt.ylabel("Pérdida")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(range(1, NUM_EPOCHS + 1), correlation_scores, label="Correlación (Prueba)")
    plt.xlabel("Épocas")
    plt.ylabel("Correlación")
    plt.legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
