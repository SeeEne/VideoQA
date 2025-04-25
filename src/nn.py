# Filename: nn.py
# Purpose: Defines the neural network architecture for probing tests.

import torch
import torch.nn as nn

class LinearProbe(nn.Module):
    """
    A simple linear classifier probe.

    Takes pre-computed feature embeddings as input and maps them to
    class logits using a single fully connected layer.
    """
    def __init__(self, embedding_dim: int, num_classes: int):
        """
        Initializes the LinearProbe.

        Args:
            embedding_dim (int): The dimensionality of the input feature embeddings
                                 (e.g., 768 for ViT-Base, 512 for CLIP ViT-Base).
            num_classes (int): The number of output classes for the specific
                               probing task (e.g., 10 for EuroSAT, 100 for CIFAR-100).
        """
        super().__init__() # Initialize the parent nn.Module class
        self.embedding_dim = embedding_dim
        self.num_classes = num_classes

        # Define the single linear layer
        self.classifier = nn.Linear(embedding_dim, num_classes)

        print(f"Initialized LinearProbe: InputDim={embedding_dim}, OutputClasses={num_classes}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the forward pass of the probe.

        Args:
            x (torch.Tensor): A batch of input embeddings with shape
                              [batch_size, embedding_dim].

        Returns:
            torch.Tensor: The output logits with shape [batch_size, num_classes].
                          (Softmax is typically applied by the loss function, e.g., CrossEntropyLoss).
        """
        # Ensure input is float32, as linear layers expect this
        if x.dtype != torch.float32:
            x = x.to(torch.float32)

        # Pass embeddings through the linear layer
        logits = self.classifier(x)
        return logits

# --- Optional: Example Instantiation (for basic check if run directly) ---
if __name__ == '__main__':
    print("\n--- Testing nn.py Basic Instantiation ---")

    # Example parameters
    example_embedding_dim = 768 # e.g., forc ViT-Base or DINOv2-Base
    example_num_classes = 10    # e.g., for EuroSAT

    print(f"\nCreating LinearProbe with embed_dim={example_embedding_dim}, num_classes={example_num_classes}")
    probe_model = LinearProbe(embedding_dim=example_embedding_dim, num_classes=example_num_classes)
    print("\nModel Structure:")
    print(probe_model)

    # Test forward pass with dummy data
    print("\nTesting forward pass...")
    batch_size = 4
    # Create a dummy batch of embeddings
    dummy_embeddings = torch.randn(batch_size, example_embedding_dim)
    print(f"Input shape: {dummy_embeddings.shape}")

    try:
        with torch.no_grad(): # No need for gradients here
            output_logits = probe_model(dummy_embeddings)
        print(f"Output logits shape: {output_logits.shape}") # Should be [batch_size, num_classes]
        print("Forward pass successful.")
    except Exception as e:
        print(f"Error during forward pass test: {e}")

    print("\n--- nn.py Test Complete ---")