from dataclasses import dataclass
from typing import Dict, Any

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
    
    
def main():
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
    tokenized_sequence_with_specials = tokenizer.add_special_tokens(tokenized_sequence)

    # Pad the tokenized sequence to a fixed length (for example, 20)
    padded_sequence = tokenizer.pad_sequence(tokenized_sequence_with_specials, 20)

    print(padded_sequence)