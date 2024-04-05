import glob
import gzip

import pandas as pd

from kmer_generation import transform_selected_genomes_df

"""
This script uses another transformation of the DNA. Instead of using the kmer frequency, we use the all DNA sequences 
that we split into tokens of same length. 
"""


def extract_sequence_from_gzipped_fasta(gzipped_fasta_file: str):
    sequence = ""
    with gzip.open(gzipped_fasta_file, "rt") as f:
        # Skip the header line
        next(f)
        # Read the sequence
        for line in f:
            sequence += line.strip()
    return sequence


def process_sequences(path_to_fasta: str = "../../../microbiodata/fasta") -> pd.DataFrame:
    seq = {}
    selected_genomes_df = pd.read_csv("../../selected_genomes.csv")
    selected_genomes_df = transform_selected_genomes_df(
        selected_genomes_df=selected_genomes_df)
    for i, genome in enumerate(glob.glob(f'{path_to_fasta}/*fna.gz')):
        if i > 20:
            break
        print(genome)
        split_genome = genome.split("/")
        if len(split_genome) > 0:
            genome_adj = genome.split("/")[-1]
        else:
            genome_adj = genome
        family_associated = selected_genomes_df[selected_genomes_df["ftp_id"] == genome_adj]["family"].mode(
        ).values[0]
        genome_sequence = extract_sequence_from_gzipped_fasta(genome)
        seq[genome_sequence] = family_associated
    seq_df = pd.DataFrame.from_dict(seq, orient="index").reset_index(drop=False).rename(columns={
        "index": "sequence",
        0: "family"
    })
    return seq_df


def split_seq_into_non_overlapping_str(sequence: str, k: int = 4) -> str:
    """
    Splits a given DNA sequence into non-overlapping words of length k and joins them into a string separated by spaces.

    Parameters:
    - sequence (str): The DNA sequence to be split.
    - k (int): The length of each word.

    Returns:
    - str: A string of non-overlapping words of length k from the sequence, separated by spaces.
    """
    # Split the sequence into non-overlapping words and join with space
    spaced_sequence = ' '.join([sequence[i:i+k]
                               for i in range(0, len(sequence), k)])

    return spaced_sequence


def split_dna_into_words(seq_df: pd.DataFrame):
    seq_df['spaced_sequence'] = seq_df['sequence'].apply(
        lambda x: split_seq_into_non_overlapping_str(x, k=8))


def quick_stats_on_df(seq_df: pd.DataFrame):
    seq_df["number_of_characters"] = seq_df["sequence"].apply(lambda x: len(x))
    seq_df["number_words"] = seq_df["spaced_sequence"].apply(
        lambda x: len(x.split(" ")))
