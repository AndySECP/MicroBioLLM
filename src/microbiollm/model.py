import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class DNAEmbedding(nn.Module):
    """
    Embed DNA sequences into a high-dimensional space.
    """

    def __init__(self, vocab_size, embed_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)

    def forward(self, x):
        return self.embedding(x)


class SequenceSSM(nn.Module):
    """
    An adaptation of the SS2D module for sequence data.
    Uses 1D convolution to model dependencies in sequences.
    """

    def __init__(self, channels, kernel_size=3):
        super().__init__()
        self.conv1d = nn.Conv1d(in_channels=channels, out_channels=channels,
                                kernel_size=kernel_size, padding=(kernel_size - 1) // 2, groups=channels)
        self.act = nn.ReLU()

    def forward(self, x):
        # x: [batch_size, seq_len, channels] -> [batch_size, channels, seq_len] for Conv1D
        x = x.permute(0, 2, 1)
        x = self.conv1d(x)
        x = self.act(x)
        # Return to original dimension order: [batch_size, seq_len, channels]
        return x.permute(0, 2, 1)


class SequenceAttentionLayer(nn.Module):
    """
    A sequence-specific attention layer.
    Implements a simple scaled dot-product attention mechanism.
    """

    def __init__(self, embed_dim):
        super().__init__()
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.scale = embed_dim ** -0.5

    def forward(self, x):
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)
        # Scaled Dot-Product Attention
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_output = torch.matmul(attention_probs, V)
        return attention_output


class SequenceTransformerBlock(nn.Module):
    """
    A Transformer block adapted for sequence data.
    Combines the SequenceSSM for local sequence modeling with an attention layer for global dependencies.
    """

    def __init__(self, embed_dim, channels, kernel_size=3):
        super().__init__()
        self.ssm = SequenceSSM(channels=channels, kernel_size=kernel_size)
        self.attention = SequenceAttentionLayer(embed_dim=embed_dim)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.ReLU(),
            nn.Linear(4 * embed_dim, embed_dim)
        )

    def forward(self, x):
        # Local sequence modeling
        x = x + self.ssm(self.norm1(x))
        # Global sequence attention
        x = x + self.attention(self.norm2(x))
        # Feed-forward network
        x = x + self.feed_forward(x)
        return x


class SequenceClassifier(nn.Module):
    """
    The main model class for sequence classification using the adapted Mamba architecture.
    Designed to classify sequences, such as DNA sequences, into predefined categories.
    """

    def __init__(self, num_classes, seq_len, embed_dim, vocab_size, depths, dims):
        super().__init__()
        self.embedder = DNAEmbedding(vocab_size, embed_dim)
        self.transformer_blocks = nn.ModuleList()

        for i in range(len(depths)):
            layer_dim = dims[i]
            for _ in range(depths[i]):
                self.transformer_blocks.append(
                    SequenceTransformerBlock(
                        embed_dim=layer_dim, channels=layer_dim)
                )
            if i < len(depths) - 1:  # Add downsampling between stages, if needed
                # Adjust this to your downsampling strategy; simple linear projection shown here
                self.transformer_blocks.append(nn.Linear(dims[i], dims[i+1]))

        self.head = nn.Linear(dims[-1], num_classes)

    def forward(self, x):
        x = self.embedder(x)
        for block in self.transformer_blocks:
            x = block(x)

        # Global average pooling across the sequence dimension
        x = x.mean(dim=1)
        x = self.head(x)
        return x
