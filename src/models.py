import torch
import numpy as np


class SimilarityModel_Cosine(torch.nn.Module):
    """
    A PyTorch module for computing similarity scores using cosine similarity.

    This module takes BERT embeddings as input and computes the cosine similarity between
    two vectors to produce a similarity score.

    Args:
        model (torch.nn.Module): A BERT model or any other model that produces embeddings
            with a dimension of 768.

    Attributes:
        bert (torch.nn.Module): The input model that produces embeddings.
        fc1 (torch.nn.Linear): A fully connected layer to transform the input embeddings.
        cosine (torch.nn.CosineSimilarity): Cosine similarity module.
        fc2 (torch.nn.Linear): A fully connected layer to produce the final similarity score.

    Methods:
        forward(input_ids, attention_mask):
            Perform the forward pass of the model.

            Args:
                input_ids (torch.Tensor): Tensor of input IDs for the BERT model.
                attention_mask (torch.Tensor): Tensor of attention masks for the BERT model.

            Returns:
                torch.Tensor: A tensor containing the similarity score.
    """

    def __init__(self, model):
        """
        Initialize the SimilarityModel_Cosine.

        Args:
            model (torch.nn.Module): A BERT model or any other model that produces embeddings
                with a dimension of 768.
        """
        super(SimilarityModel_Cosine, self).__init__()
        self.bert = model
        self.fc1 = torch.nn.Linear(768, 128)  # 768 is the output dimension of BERT
        self.cosine = torch.nn.CosineSimilarity()
        self.fc2 = torch.nn.Linear(128, 1)

    def forward(self, input_ids, attention_mask):
        """
        Perform the forward pass of the model.

        Args:
            input_ids (torch.Tensor): Tensor of input IDs for the BERT model.
            attention_mask (torch.Tensor): Tensor of attention masks for the BERT model.

        Returns:
            torch.Tensor: A tensor containing the similarity score.
        """
        outputs = self.bert(input_ids, attention_mask)
        pooled_output = outputs["last_hidden_state"][:, 0]
        x = self.fc1(pooled_output)
        x = self.cosine(x)
        score = self.fc2(x)
        return score


class SimilarityModel_None(torch.nn.Module):
    """
    A PyTorch neural network model for computing similarity scores using BERT embeddings.

    This model takes BERT embeddings as input and computes a similarity score between input pairs.

    Args:
        model (torch.nn.Module): A BERT model or any model producing BERT-style embeddings.

    Attributes:
        bert (torch.nn.Module): The underlying BERT model.
        fc1 (torch.nn.Linear): The first fully connected layer with input size 768 (BERT output dimension) and output size 128.
        fc2 (torch.nn.Linear): The second fully connected layer with input size 128 and output size 1.

    Methods:
        forward(input_ids, attention_mask):
            Computes the similarity score between input pairs using BERT embeddings.
    """
    def __init__(self, model):
        """
        Initialize the SimilarityModel_None.

        Args:
            model (torch.nn.Module): A BERT model or any model producing BERT-style embeddings.
        """
        super(SimilarityModel_None, self).__init__()
        self.bert = model
        self.fc1 = torch.nn.Linear(768, 128)  # 768 is the output dimension of BERT
        self.fc2 = torch.nn.Linear(128, 1)

    def forward(self, input_ids, attention_mask):
        """
        Forward pass of the model to compute similarity scores.

        Args:
            input_ids (torch.Tensor): Tensor containing input IDs.
            attention_mask (torch.Tensor): Tensor containing attention masks.

        Returns:
            torch.Tensor: Tensor containing similarity scores.
        """
        outputs = self.bert(input_ids, attention_mask)
        pooled_output = outputs["last_hidden_state"][:, 0]
        x = self.fc1(pooled_output)
        score = self.fc2(x)
        return score


class SimilarityModel_ReLU(torch.nn.Module):
    """
    A PyTorch model for computing similarity scores using a ReLU activation function.

    This class takes a pre-trained model (e.g., a BERT model) and adds two fully connected layers to compute similarity scores.

    Args:
        model (torch.nn.Module): The pre-trained model (e.g., BERT) to extract features.

    Attributes:
        bert (torch.nn.Module): The pre-trained model for feature extraction.
        fc1 (torch.nn.Linear): The first fully connected layer for feature transformation.
        fc2 (torch.nn.Linear): The second fully connected layer for similarity score computation.

    Methods:
        forward(input_ids, attention_mask):
            Computes the similarity score between input sequences.
    """

    def __init__(self, model):
        """
        Initialize the SimilarityModel_ReLU.

        Args:
            model (torch.nn.Module): The pre-trained model (e.g., BERT) to extract features.
        """
        super(SimilarityModel_ReLU, self).__init__()
        self.bert = model
        self.fc1 = torch.nn.Linear(768, 128)  # 768 is the dimension of BERT's output
        self.fc2 = torch.nn.Linear(128, 1)

    def forward(self, input_ids, attention_mask):
        """
        Forward pass to compute similarity scores.

        Args:
            input_ids (torch.Tensor): Tensor of input token IDs.
            attention_mask (torch.Tensor): Tensor of input attention masks.

        Returns:
            torch.Tensor: Tensor of similarity scores computed using ReLU activation.

        Note:
            The forward method takes input token IDs and attention masks, passes them through the pre-trained model,
            and computes similarity scores using a ReLU activation function.
        """
        outputs = self.bert(input_ids, attention_mask)
        pooled_output = outputs["last_hidden_state"][:, 0]
        x = self.fc1(pooled_output)
        x = self.relu(x)  # It should be self.relu(x) instead of self.relu(x)
        score = self.fc2(x)
        return score
