import torch
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
from nltk.util import ngrams
from nltk.metrics import jaccard_distance
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, get_linear_schedule_with_warmup
from torch.optim import AdamW
from scipy.stats import pearsonr
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer


from constants import MAX_LENGTH, BATCH_SIZE, NUM_EPOCHS

def jaccard_score(text1, text2):
    set1 = set(ngrams(text1.split(), 1))
    set2 = set(ngrams(text2.split(), 1))
    return 1 - jaccard_distance(set1, set2)

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
        pd.DataFrame: A DataFrame containing the original data with additional columns for model predictions.

    This function performs training for a specified number of epochs. It uses the AdamW optimizer with a learning rate
    scheduler to update the model's parameters. Training progress is printed at each epoch, displaying the average loss.
    Additionally, it calculates and appends model predictions, Jaccard scores, and cosine similarity scores to the input data.
    The function returns a DataFrame containing the original data with added columns for model predictions, Jaccard scores,
    and cosine similarity scores for both the test and validation sets.
    """

    train_loader, test_loader, val_loader, train_df, test_df, val_df = data_process(path)

    test_df["Jaccard Score"] = [jaccard_score(row["abstract"], row["paraphrase_abstract"]) for  _, row in test_df.iterrows()]
    val_df["Jaccard Score"] = [jaccard_score(row["abstract"], row["paraphrase_abstract"]) for _, row in val_df.iterrows()]

    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix_test_original = tfidf_vectorizer.fit_transform(test_df["abstract"])
    tfidf_matrix_test_paraphrase = tfidf_vectorizer.transform(test_df["paraphrase_abstract"])
    cosine_sim_test = cosine_similarity(tfidf_matrix_test_original, tfidf_matrix_test_paraphrase)

    tfidf_matrix_val_original = tfidf_vectorizer.fit_transform(val_df["abstract"])
    tfidf_matrix_val_paraphrase = tfidf_vectorizer.transform(val_df["paraphrase_abstract"])
    cosine_sim_val = cosine_similarity(tfidf_matrix_val_original, tfidf_matrix_val_paraphrase)

    test_df["Cosine Score"] = [cosine_sim_test[0][0]] * len(test_df)
    val_df["Cosine Score"] = [cosine_sim_val[0][0]] * len(val_df)

    train_losses = []
    val_losses = []
    correlation_scores = []

    optimizer = AdamW(model.parameters(), lr=learning_rate)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=len(train_loader) * NUM_EPOCHS
    )
    
    output_df = None

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
        correlation, test_df_with_predictions = test_model(test_loader, test_df, model)
        correlation_scores.append(correlation)

        # Validate the model
        average_val_loss, val_df_with_predictions = val_model(val_loader, model, criterion, val_df)
        val_losses.append(average_val_loss)

        if output_df is None:
            output_df = pd.concat([test_df_with_predictions, val_df_with_predictions], ignore_index=True)
        else:
            output_df = pd.concat([output_df, test_df_with_predictions, val_df_with_predictions], ignore_index=True)

    plot_model(correlation_scores, train_losses, val_losses, save_path)

    return output_df


def test_model(test_loader: DataLoader, test_df: pd.DataFrame, model: torch.nn.Module):
    """
    Evaluate a PyTorch neural network model's performance on a given test dataset and return predictions.

    Parameters:
    - test_loader (DataLoader): A PyTorch DataLoader containing the test dataset.
    - test_df (pd.DataFrame): A Pandas DataFrame containing the test data, including a "Score (0-10)" column.
    - model (torch.nn.Module): A PyTorch neural network model for prediction.

    Returns:
    - correlation (float): The Pearson correlation coefficient between the model's predictions and the true scores.
    - output_df (pd.DataFrame): A DataFrame with the test data and an additional "Predicted Score" column.

    This function evaluates the model's performance on the test data by making predictions and calculating
    the Pearson correlation coefficient between the predicted scores and the true scores in the test DataFrame.
    Additionally, it returns a DataFrame with the original test data and an extra column containing the model's predictions.
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

    correlation, _ = pearsonr(predicted_scores, true_scores)
    print(f"Correlation between predictions and the 'Score' field in the test dataset: {correlation}")

    output_df = test_df.copy()
    output_df["Predicted Score"] = predicted_scores

    return correlation, output_df

def val_model(
    val_loader: DataLoader, model: torch.nn.Module, criterion: torch.nn.Module, val_df: pd.DataFrame
):
    """
    Evaluate a PyTorch neural network model's performance on a given validation dataset and return predictions.

    Parameters:
    - val_loader (DataLoader): A PyTorch DataLoader containing the validation dataset.
    - model (torch.nn.Module): A PyTorch neural network model for prediction.
    - criterion (torch.nn.Module): Loss criterion for validation.
    - val_df (pd.DataFrame): A Pandas DataFrame containing the validation data, including a "score" column.

    Returns:
    - average_val_loss (float): The average loss calculated during validation.
    - output_df (pd.DataFrame): A DataFrame with the validation data and an additional "Predicted Score" column.

    This function evaluates the model's performance on the validation data by making predictions and calculating the
    average loss using the specified loss criterion. It also returns a DataFrame with the original validation data
    and an extra column containing the model's predictions.
    """

    model.eval()
    val_loss = 0.0
    predictions = []
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
            predictions.extend(output)

    average_val_loss = val_loss / len(val_loader)

    output_df = val_df.copy()
    predicted_scores = [p.item() for p in predictions]
    output_df["Predicted Score"] = predicted_scores

    return average_val_loss, output_df


def plot_model(
    correlation_scores: list, train_losses: list, val_losses: list, save_path: str = ""
):
    """
    Plot the training progress and correlation scores of a neural network model.

    Parameters:
    - correlation_scores (list): A list of correlation scores for each epoch.
    - train_losses (list): A list of training losses for each epoch.
    - val_losses (list): A list of validation losses for each epoch.
    - save_path (str, optional): Path to save the plot as an image (default is "").

    This function creates a two-panel plot to visualize the training progress and correlation scores of a neural network model
    across epochs. The left panel displays training and validation losses over epochs, while the right panel shows the
    correlation scores over epochs. If 'save_path' is provided, the plot is saved as an image; otherwise, it is displayed.
    """

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, NUM_EPOCHS + 1), train_losses, label="Training")
    plt.plot(range(1, NUM_EPOCHS + 1), val_losses, label="Validation")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(range(1, NUM_EPOCHS + 1), correlation_scores, label="Correlation")
    plt.xlabel("Epochs")
    plt.ylabel("Correlation")
    plt.legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
