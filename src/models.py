import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from transformers import PreTrainedModel


class SimilarityModel_Cosine(nn.Module):
    """
    A PyTorch module for computing similarity scores using cosine similarity.

    This module takes BERT embeddings as input and computes the cosine similarity between
    two vectors to produce a similarity score.

    Args:
        model (PreTrainedModel): A pre-trained model (e.g., BERT) or any other model that produces embeddings
            with a dimension of 768.

    Attributes:
        bert (PreTrainedModel): The input model that produces embeddings.
        fc1 (nn.Linear): A fully connected layer to transform the input embeddings.
        cosine (nn.CosineSimilarity): Cosine similarity module.
        fc2 (nn.Linear): A fully connected layer to produce the final similarity score.

    Methods:
        forward(input_ids_abstract, attention_mask_abstract, input_ids_paraphrase, attention_mask_paraphrase):
            Perform the forward pass of the model.

            Args:
                input_ids_abstract (Tensor): Tensor of input token IDs for the abstract.
                attention_mask_abstract (Tensor): Tensor of input attention masks for the abstract.
                input_ids_paraphrase (Tensor): Tensor of input token IDs for the paraphrase.
                attention_mask_paraphrase (Tensor): Tensor of input attention masks for the paraphrase.

            Returns:
                Tensor: A tensor containing the similarity score.
    """

    def __init__(self, model: PreTrainedModel):
        """
        Initialize the SimilarityModel_Cosine.

        Args:
            model (PreTrainedModel): A pre-trained model (e.g., BERT) or any other model that produces embeddings
                with a dimension of 768.
        """
        super(SimilarityModel_Cosine, self).__init__()
        self.bert = model
        self.fc1 = nn.Linear(768, 128)  # 768 is the output dimension of BERT
        self.cosine = nn.CosineSimilarity(dim=1)
        self.fc2 = nn.Linear(128, 1)

    def forward(
        self,
        input_ids_abstract: Tensor,
        attention_mask_abstract: Tensor,
        input_ids_paraphrase: Tensor,
        attention_mask_paraphrase: Tensor,
    ) -> Tensor:
        """
        Forward pass to compute similarity scores using cosine similarity.

        Args:
            input_ids_abstract (Tensor): Tensor of input token IDs for the abstract.
            attention_mask_abstract (Tensor): Tensor of input attention masks for the abstract.
            input_ids_paraphrase (Tensor): Tensor of input token IDs for the paraphrase.
            attention_mask_paraphrase (Tensor): Tensor of input attention masks for the paraphrase.

        Returns:
            Tensor: Tensor of similarity scores computed using cosine similarity.

        Note:
            The forward method takes input token IDs and attention masks, passes them through the pre-trained model,
            and computes similarity scores using cosine similarity.
        """
        outputs_abstract = self.bert(input_ids_abstract, attention_mask_abstract)
        pooled_output_abstract = outputs_abstract["last_hidden_state"][:, 0]

        outputs_paraphrase = self.bert(input_ids_paraphrase, attention_mask_paraphrase)
        pooled_output_paraphrase = outputs_paraphrase["last_hidden_state"][:, 0]

        x_abstract = self.fc1(pooled_output_abstract)
        x_paraphrase = self.fc1(pooled_output_paraphrase)

        x_abstract = F.normalize(x_abstract, dim=1)  # Normalize the features
        x_paraphrase = F.normalize(x_paraphrase, dim=1)  # Normalize the features

        similarity_score = self.cosine(
            x_abstract, x_paraphrase
        )  # Compute cosine similarity
        score = self.fc2(similarity_score.view(-1, 1))
        return score


class SimilarityModel_None(nn.Module):
    """
    A PyTorch neural network model for computing similarity scores using BERT embeddings.

    This model takes BERT embeddings as input and computes a similarity score between input pairs.

    Args:
        model (PreTrainedModel): A pre-trained model (e.g., BERT) or any model producing BERT-style embeddings.

    Attributes:
        bert (PreTrainedModel): The underlying BERT model.
        fc1 (nn.Linear): The first fully connected layer with input size 768 (BERT output dimension) and output size 128.
        fc2 (nn.Linear): The second fully connected layer with input size 128 and output size 1.

    Methods:
        forward(input_ids_abstract, attention_mask_abstract, input_ids_paraphrase, attention_mask_paraphrase):
            Computes the similarity score between input pairs using BERT embeddings.
    """

    def __init__(self, model: PreTrainedModel):
        """
        Initialize the SimilarityModel_None.

        Args:
            model (PreTrainedModel): A pre-trained model (e.g., BERT) or any model producing BERT-style embeddings.
        """
        super(SimilarityModel_None, self).__init__()
        self.bert = model
        self.fc1 = nn.Linear(768, 128)  # 768 is the output dimension of BERT
        self.fc2 = nn.Linear(128, 1)

    def forward(
        self,
        input_ids_abstract: Tensor,
        attention_mask_abstract: Tensor,
        input_ids_paraphrase: Tensor,
        attention_mask_paraphrase: Tensor,
    ) -> Tensor:
        """
        Forward pass of the model to compute similarity scores.

        Args:
            input_ids_abstract (Tensor): Tensor containing input IDs for the abstract.
            attention_mask_abstract (Tensor): Tensor containing attention masks for the abstract.
            input_ids_paraphrase (Tensor): Tensor containing input IDs for the paraphrase.
            attention_mask_paraphrase (Tensor): Tensor containing attention masks for the paraphrase.

        Returns:
            Tensor: Tensor containing similarity scores.
        """
        outputs_abstract = self.bert(input_ids_abstract, attention_mask_abstract)
        pooled_output_abstract = outputs_abstract["last_hidden_state"][:, 0]

        outputs_paraphrase = self.bert(input_ids_paraphrase, attention_mask_paraphrase)
        pooled_output_paraphrase = outputs_paraphrase["last_hidden_state"][:, 0]

        x_abstract = self.fc1(pooled_output_abstract)
        x_paraphrase = self.fc1(pooled_output_paraphrase)

        score = self.fc2(
            x_abstract - x_paraphrase
        )  # Compute similarity score as the difference
        return score


class SimilarityModel_ReLU(nn.Module):
    """
    A PyTorch model for computing similarity scores using a ReLU activation function.

    This class takes a pre-trained model (e.g., a BERT model) and adds two fully connected layers to compute similarity scores.

    Args:
        model (PreTrainedModel): The pre-trained model (e.g., BERT) to extract features.

    Attributes:
        bert (PreTrainedModel): The pre-trained model for feature extraction.
        fc1 (nn.Linear): The first fully connected layer for feature transformation.
        fc2 (nn.Linear): The second fully connected layer for similarity score computation.

    Methods:
        forward(input_ids_abstract, attention_mask_abstract, input_ids_paraphrase, attention_mask_paraphrase):
            Computes the similarity score between input sequences.
    """

    def __init__(self, model: PreTrainedModel):
        """
        Initialize the SimilarityModel_ReLU.

        Args:
            model (PreTrainedModel): The pre-trained model (e.g., BERT) to extract features.
        """
        super(SimilarityModel_ReLU, self).__init__()
        self.bert = model
        self.fc1 = nn.Linear(768, 128)  # 768 is the dimension of BERT's output
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 1)

    def forward(
        self,
        input_ids_abstract: Tensor,
        attention_mask_abstract: Tensor,
        input_ids_paraphrase: Tensor,
        attention_mask_paraphrase: Tensor,
    ) -> Tensor:
        """
        Forward pass to compute similarity scores.

        Args:
            input_ids_abstract (Tensor): Tensor of input token IDs for the abstract.
            attention_mask_abstract (Tensor): Tensor of input attention masks for the abstract.
            input_ids_paraphrase (Tensor): Tensor of input token IDs for the paraphrase.
            attention_mask_paraphrase (Tensor): Tensor of input attention masks for the paraphrase.

        Returns:
            Tensor: Tensor of similarity scores computed using ReLU activation.

        Note:
            The forward method takes input token IDs and attention masks, passes them through the pre-trained model,
            and computes similarity scores using a ReLU activation function.
        """
        outputs_abstract = self.bert(input_ids_abstract, attention_mask_abstract)
        pooled_output_abstract = outputs_abstract["last_hidden_state"][:, 0]

        outputs_paraphrase = self.bert(input_ids_paraphrase, attention_mask_paraphrase)
        pooled_output_paraphrase = outputs_paraphrase["last_hidden_state"][:, 0]

        x_abstract = self.fc1(pooled_output_abstract)
        x_abstract = self.relu(x_abstract)

        x_paraphrase = self.fc1(pooled_output_paraphrase)
        x_paraphrase = self.relu(x_paraphrase)

        score = self.fc2(
            x_abstract - x_paraphrase
        )  # Compute similarity score using ReLU
        return score
