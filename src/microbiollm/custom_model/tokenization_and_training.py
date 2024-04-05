import pandas as pd
import numpy as np
import torch
import itertools
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
from model import SequenceClassifier

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer

# Preparing dataset


class DNASequenceDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = sequences
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        label = self.labels[idx]
        return torch.tensor(sequence, dtype=torch.long), torch.tensor(label, dtype=torch.long)


def bpe_encoding():
    # Initialize a tokenizer with the BPE model
    tokenizer = Tokenizer(BPE(unk_token="<unk>"))

    # Use byte-level pre-tokenization to process DNA sequences (treats each character as a separate token initially)
    tokenizer.pre_tokenizer = ByteLevel()

    # Initialize the trainer, specify the special tokens and the desired vocabulary size
    trainer = BpeTrainer(
        special_tokens=["<pad>", "<unk>", "<s>", "</s>"], vocab_size=10000)

    # Train the tokenizer on your file
    tokenizer.train(['dna_sequences.txt'], trainer)


def build_4_mer_vocab() -> dict:
    # Initialize vocab with <pad> and <unk>
    vocab = {"<pad>": 0, "<unk>": 1}

    # Generate all 4-mers and add them to the vocab
    nucleotides = ['A', 'C', 'G', 'T']
    four_mers = [''.join(p) for p in itertools.product(nucleotides, repeat=4)]
    for four_mer in four_mers:
        vocab[four_mer] = len(vocab)
    return vocab


def processing_sequences(df: pd.DataFrame):

    # Tokenizing sequences
    vocab = build_4_mer_vocab()

    def tokenize_sequence(sequence, vocab):
        tokens = [vocab.get(word, vocab["<unk>"]) for word in sequence.split()]
        return np.array(tokens, dtype=np.int64)

    df['tokenized_sequences'] = df['spaced_sequence'].apply(
        lambda seq: tokenize_sequence(seq, vocab=vocab))

    # Label encoding for the families
    label_encoder = LabelEncoder()
    df['family_encoded'] = label_encoder.fit_transform(df['family'])

    X_train, X_test, y_train, y_test = train_test_split(
        df['tokenized_sequences'].tolist(), df['family_encoded'].tolist(), test_size=0.2)

    train_dataset = DNASequenceDataset(X_train, y_train)
    test_dataset = DNASequenceDataset(X_test, y_test)

    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    return train_dataloader, test_dataloader, label_encoder, X_train, y_train, vocab


def evaluate_model(model, dataloader, device):
    model.eval()  # Set the model to evaluation mode
    total, correct = 0, 0

    with torch.no_grad():
        for sequences, labels in dataloader:
            sequences, labels = sequences.to(device), labels.to(device)
            outputs = model(sequences)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f'Accuracy of the model on the test sequences: {accuracy:.2f}%')


def train_model(df: pd.DataFrame, epochs=10):
    # Prepare data and model
    train_dataloader, test_dataloader, label_encoder, X_train, y_train, vocab = processing_sequences(
        df=df)
    model = SequenceClassifier(num_classes=len(label_encoder.classes_), seq_len=max(
        [len(seq) for seq in X_train]), embed_dim=128, vocab_size=len(vocab), depths=[2, 2], dims=[128, 128])
    optimizer = torch.optim.Adam(model.parameters())
    loss_fn = torch.nn.CrossEntropyLoss()

    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    for epoch in range(epochs):
        model.train()  # Set model to training mode
        total_loss = 0

        for sequences, labels in train_dataloader:
            sequences, labels = sequences.to(device), labels.to(device)

            # Forward pass
            outputs = model(sequences)
            loss = loss_fn(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_dataloader)
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")

        # Evaluate the model with the test dataset
        evaluate_model(model, test_dataloader, device)
