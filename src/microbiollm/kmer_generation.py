#! /usr/bin/env python3

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import glob
import gzip
import logging
import sys

import pandas as pd
from transformers import AutoTokenizer
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split

"""
This script generates the kmer frequencies from the selected genomes (downloaded using the data_base_creation.py script)
and will tokenize the data (we used here the Mistral tokenizer but that might not be the best option moving forward) and
represent each token with a vector 
Example use: 
python3 MicroBioLLM/src/microbiollm/kmer_generation.py -t -s -g database
"""

logging.basicConfig(level=logging.INFO,
                    format='[%(filename)s:%(lineno)s - %(funcName)s()] [%(asctime)s %(levelname)s] %(message)s')
baselog = logging.getLogger(__name__)
baselog.setLevel(logging.INFO)


def extract_sequence_from_gzipped_fasta(gzipped_fasta_file: str) -> str:
    sequence = ""
    with gzip.open(gzipped_fasta_file, "rt") as f:
        # Skip the header line
        next(f)
        # Read the sequence
        for line in f:
            sequence += line.strip()
    return sequence


def count_kmer_types(sequence: str, k: int) -> dict:
    kmer_counts = {}
    for i in range(len(sequence) - k + 1):
        kmer = sequence[i:i + k]
        if any(base not in "ATCG" for base in kmer):
            pass
        else:
            if kmer in kmer_counts:
                kmer_counts[kmer] += 1
            else:
                kmer_counts[kmer] = 1
    return kmer_counts


def create_kmer_dataframe(sequence_data: dict, k: int):
    # List to store each row's data
    rows_list = []

    for sequence, taxid in sequence_data.items():
        # Count the k-mers in the sequence
        kmer_counts = count_kmer_types(sequence, k)

        # Add the taxid as part of the row data
        row_data = kmer_counts
        row_data['taxid'] = taxid

        # Append this row's data to the rows_list
        rows_list.append(row_data)

    # Convert the list of dictionaries to DataFrame
    df = pd.DataFrame(rows_list).set_index('taxid').fillna(0).astype(int)

    return df


def generate_kmer_sentences_with_loop(sequence_data: dict, kmer_size: int = 8) -> dict:
    """
    Generate k-mer sentences from DNA sequences.
    """
    kmer_sentences = {}
    for sequence, label in sequence_data.items():
        print(label)
        kmer_list = [sequence[i:i + kmer_size]
                     for i in range(len(sequence) - kmer_size + 1)]
        sentence_kmer = " ".join(kmer_list)
        kmer_sentences[sentence_kmer] = label
    return kmer_sentences


def generate_kmer_sentence(sequence: str, kmer_size: int):
    """
    Generate a k-mer sentence from a single DNA sequence.
    """
    kmer_list = [sequence[i:i + kmer_size]
                 for i in range(len(sequence) - kmer_size + 1)]
    return " ".join(kmer_list)


def parallel_generate_kmer_sentences(sequence_data: dict, kmer_size: int, max_workers: int = 10):
    """
    Parallel generation of k-mer sentences from DNA sequences.
    """
    kmer_sentences = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_seq = {executor.submit(
            generate_kmer_sentence, seq, kmer_size): seq for seq in sequence_data.keys()}
        for future in as_completed(future_to_seq):
            seq = future_to_seq[future]
            try:
                kmer_sentence = future.result()
            except Exception as exc:
                print('%r generated an exception: %s' % (seq, exc))
            else:
                kmer_sentences[kmer_sentence] = sequence_data[seq]
    return kmer_sentences


def tokenize_and_prepare_datasets(sequence_data: dict, kmer_size: int,
                                  tokenizer_model="mistralai/Mistral-7B-Instruct-v0.2"):
    """
    Tokenize k-mer sentences and prepare datasets for training/testing.
    """
    # First, generate k-mer sentences

    kmer_sentences = parallel_generate_kmer_sentences(sequence_data, kmer_size)

    print("kmer_sentence_done!")

    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token if tokenizer.eos_token else '[PAD]'

    # Prepare data for tokenization
    sentences, labels = zip(*kmer_sentences.items())

    # Tokenize sentences
    encodings = tokenizer(list(sentences), truncation=True, padding=True)

    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(
        encodings['input_ids'], labels, test_size=0.1, random_state=42)

    # Create datasets
    train_dataset = Dataset.from_dict({
        "input_ids": X_train,
        "attention_mask": encodings['attention_mask'][:len(X_train)],
        "labels": y_train
    })
    test_dataset = Dataset.from_dict({
        "input_ids": X_test,
        "attention_mask": encodings['attention_mask'][len(X_train):],
        "labels": y_test
    })

    tokenized_datasets = DatasetDict({
        'train': train_dataset,
        'test': test_dataset
    })

    return tokenized_datasets


def transform_selected_genomes_df(selected_genomes_df: pd.DataFrame):
    """
    Will add the name of the fasta in the selected genome dataframe
    :param selected_genomes_df:
    :return:
    """
    selected_genomes_sub = selected_genomes_df[["taxid", "species_taxid", "organism_name", "family", "ftp_path"]]
    selected_genomes_sub["ftp_id"] = selected_genomes_sub["ftp_path"].str.split("/")
    selected_genomes_sub["ftp_id"] = selected_genomes_sub["ftp_id"].apply(lambda x: x[-1] if len(x) > 0 else "")
    selected_genomes_sub["ftp_id"] = selected_genomes_sub["ftp_id"] + "_genomic.fna.gz"
    return selected_genomes_sub


def generate_kmer(args):
    logger = baselog.getChild('Generate kmers')
    seq = {}
    selected_genomes_df = pd.read_csv(f"{args.genome_dir}/selected_organisms.csv")
    selected_genomes_df = transform_selected_genomes_df(selected_genomes_df=selected_genomes_df)
    for i, genome in enumerate(glob.glob(f'{args.genome_dir}/*fna.gz')):
        if i > 20:
            break
        logger.info(f"Processing {genome}")
        split_genome = genome.split("/")
        if len(split_genome) > 0:
            genome_adj = genome.split("/")[-1]
        else:
            genome_adj = genome
        family_associated = selected_genomes_df[selected_genomes_df["ftp_id"] == genome_adj]["family"].mode(
        ).values[0]
        genome_sequence = extract_sequence_from_gzipped_fasta(genome)
        seq[genome_sequence] = family_associated
    if args.tokenize_bool:
        tokenized_datasets = tokenize_and_prepare_datasets(
            sequence_data=seq, kmer_size=4, tokenizer_model="mistralai/Mistral-7B-Instruct-v0.2")
        tokenized_datasets.save_to_disk("path/to/save/tokenized_datasets")

    df_seq = create_kmer_dataframe(sequence_data=seq, k=4)

    if args.save_matrix_to_csv:
        df_seq.to_csv("sequence_matrix.csv")

    return df_seq


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='kmer generation')
    parser.add_argument('-t', '--tokenize_bool', dest='tokenize_bool', help='tokenize_bool',action='store_true',
                        required=True)
    parser.add_argument('-s', '--save_matrix_to_csv', dest='save_matrix_to_csv', help='save_matrix_to_csv',
                        action='store_true', required=True)
    parser.add_argument('-g', '--genome_dir', dest='genome_dir', help='directory to store the genomes', required=True)

    args = parser.parse_args()

    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

    generate_kmer(args)
