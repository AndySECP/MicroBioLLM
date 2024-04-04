#! /usr/bin/env python3

from dataclasses import dataclass
import json
import logging


import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer, AutoTokenizer, DataCollatorWithPaddingm, BitsAndBytesConfig
from datasets import Dataset, DatasetDict
import torch

logger = logging.getLogger(__name__)

"""
This script tokenized the kmers and train and fine tuned the Mistral LLM 
"""

class KmerCountDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = sequences
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids": torch.tensor(self.sequences[idx], dtype=torch.long),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long)
        }


@dataclass
class MicroBioLLMConfig:
    kmer_size: int
    n_tokens: int
    n_special_tokens: int
    pad_token_id: int = 0
    eos_token_id: int = 1
    unk_token_id: int = 2

    def __post_init__(self):
        self.vocab_size = self.n_tokens + self.n_special_tokens
        assert self.pad_token_id < self.n_special_tokens, "Special token IDs must be within the special tokens range"
        assert self.eos_token_id < self.n_special_tokens, "Special token IDs must be within the special tokens range"
        assert self.unk_token_id < self.n_special_tokens, "Special token IDs must be within the special tokens range"


class MicroBioTokenizer:
    def __init__(self, config: MicroBioLLMConfig) -> None:
        self.config = config
        self.kmer_to_token = {}
        self.token_to_kmer = {}

    def build_vocab(self, sequences: list):
        unique_kmers = set()
        for sequence in sequences:
            for i in range(len(sequence) - self.config.kmer_size + 1):
                kmer = sequence[i:i + self.config.kmer_size]
                unique_kmers.add(kmer)

        for i, kmer in enumerate(unique_kmers, start=self.config.n_special_tokens):
            self.kmer_to_token[kmer] = i
            self.token_to_kmer[i] = kmer

    def tokenize_sequence(self, sequence: str):
        tokens = []
        for i in range(len(sequence) - self.config.kmer_size + 1):
            kmer = sequence[i:i + self.config.kmer_size]
            token = self.kmer_to_token.get(kmer, self.config.unk_token_id)
            tokens.append(token)
        return tokens

    def add_special_tokens(self, tokens: list):
        return [self.config.eos_token_id] + tokens + [self.config.eos_token_id]

    def pad_sequence(self, tokens: list, max_length: int):
        padded_length = max_length - len(tokens)
        return tokens + [self.config.pad_token_id] * padded_length


class MicroBioFewShots:
    def __init__():
        pass


def prepare_data_and_tokenize_string_input(data: dict, tokenizer_model="mistralai/Mistral-7B-Instruct-v0.2"):
    # Load the tokenizer
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_model, quantization_config=bnb_config, device_map="auto")

    # Check if the tokenizer has a pad token, if not, set it to the eos_token
    # Ensure the tokenizer has a pad token set
    if tokenizer.pad_token is None:
        if tokenizer.eos_token:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            # If there is no eos token, explicitly add a pad token.
            # This is necessary for models that require a pad token for proper batching.
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    # Your existing code for tokenization
    X = list(data.keys())
    # It seems you intended to use keys as labels, adjust according to your actual needs
    y = list(data.values())

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.1, random_state=42)

    # Tokenize with padding and truncation
    train_encodings = tokenizer(X_train, truncation=True, padding=True)
    test_encodings = tokenizer(X_test, truncation=True, padding=True)

    # Create Hugging Face datasets
    train_dataset = Dataset.from_dict(
        {"input_ids": train_encodings['input_ids'], "attention_mask": train_encodings['attention_mask'], "labels": y_train})
    test_dataset = Dataset.from_dict(
        {"input_ids": test_encodings['input_ids'], "attention_mask": test_encodings['attention_mask'], "labels": y_test})

    tokenized_datasets = DatasetDict({
        'train': train_dataset,
        'test': test_dataset
    })

    return tokenized_datasets


class MicroBioFineTuning:
    def __init__(self, tokenized_datasets):
        self.tokenized_datasets = tokenized_datasets
        self.model = None
        self.trainer = None

    def model_train(self):
        # tokenizer = AutoTokenizer.from_pretrained(
        #    "mistralai/Mistral-7B-Instruct-v0.2")
        self.model = AutoModelForSequenceClassification.from_pretrained(
            "mistralai/Mistral-7B-Instruct-v0.2", num_labels=2)

        training_args = TrainingArguments(
            output_dir="./results",
            evaluation_strategy="epoch",
            learning_rate=2e-5,
            per_device_train_batch_size=4,
            per_device_eval_batch_size=4,
            num_train_epochs=3,
            weight_decay=0.01,
            logging_steps=10,
            # fp16=True,
        )
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.tokenized_datasets["train"],
            eval_dataset=self.tokenized_datasets["test"],
        )
        self.trainer.train()
        self.model.save_pretrained('./fine_tuned_model')

        train_metrics = self.trainer.evaluate(
            eval_dataset=self.tokenized_datasets["train"])
        test_metrics = self.trainer.evaluate(
            eval_dataset=self.tokenized_datasets["test"])
        with open("./training_metrics.json", "w") as tm:
            json.dump(train_metrics, tm)
        with open("./test_metrics.json", "w") as tm:
            json.dump(test_metrics, tm)

    def model_predict(self, texts):
        # Assuming `texts` is a list of sequences to predict
        tokenizer = AutoTokenizer.from_pretrained(
            "mistralai/Mistral-7B-Instruct-v0.2")
        tokens = tokenizer(texts, padding=True,
                           truncation=True, return_tensors="pt")
        predictions = self.model(**tokens)
        return predictions.logits.argmax(dim=-1)


def main_0():
    config = MicroBioLLMConfig(kmer_size=8, n_tokens=1000, n_special_tokens=3)
    tokenizer = MicroBioTokenizer(config)

    # Example sequence data
    sequences = [
        "ATCGATCGATCGATCGATCGATCGATCGATCGATCG",
        "GCTAGCTAGCTA",
        # Add more sequences as needed
    ]

    # Build the vocabulary from the sequences
    tokenizer.build_vocab(sequences)

    # Tokenize a sequence and add special tokens
    tokenized_sequence = tokenizer.tokenize_sequence(sequences[0])
    tokenized_sequence_with_specials = tokenizer.add_special_tokens(
        tokenized_sequence)

    # Pad the tokenized sequence to a fixed length (for example, 20)
    padded_sequence = tokenizer.pad_sequence(
        tokenized_sequence_with_specials, 20)

    logger.info(padded_sequence)


def dtf_to_sentence(dtf):
    """
    Converting a given dtf to sentences describing the frequency of each k-mers.
    """
    # Initialize an empty dictionary to store the result
    sentences = {}

    # Pre-compute the list of 3-mers to avoid repeatedly accessing the dictionary's keys
    kmers = [k for k in dtf.columns if k != "label"]

    # Iterate over indices and taxa simultaneously for direct access
    for index, taxon in enumerate(dtf["label"]):
        # Use a list comprehension for concise and potentially faster execution
        sentence = ' '.join(
            f"{kmer} {dtf.loc[index, kmer]}" for kmer in kmers if dtf.loc[index, kmer] > 0)

        # Assign the sentence to the taxon in the result dictionary
        sentences[sentence] = taxon

    return sentences


def main():
    logger.info("start!")

    # Creating the DataFrame
    # df = pd.DataFrame(data).set_index("tax").T
    df = pd.read_csv("../../sequence_matrix.csv")
    df_formated = df.reset_index(drop=False).rename(columns={"taxid": "label"})
    sentences = dtf_to_sentence(dtf=df_formated)
    logger.info("conversion to sentence done!")
    # dict_data = {}
    tokenized_datasets = prepare_data_and_tokenize_string_input(
        data=sentences)
    logger.info("tokenized_datasets done!")
    MBFT = MicroBioFineTuning(
        tokenized_datasets=tokenized_datasets
    )
    MBFT.model_train()
    logger.info("done!")


if __name__ == "__main__":
    main()
