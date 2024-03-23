import glob
import gzip

import pandas as pd


def extract_sequence_from_gzipped_fasta(gzipped_fasta_file):
    sequence = ""
    with gzip.open(gzipped_fasta_file, "rt") as f:
        # Skip the header line
        next(f)
        # Read the sequence
        for line in f:
            sequence += line.strip()
    return sequence


def count_kmer_types(sequence, k):
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


def create_kmer_dataframe(sequence_data, k):
    data = {}
    for taxid, sequence in sequence_data.items():
        print(taxid)
        kmer_counts = count_kmer_types(sequence, k)
        data[taxid] = kmer_counts

    # Convert data dictionary to DataFrame
    df = pd.DataFrame(data).fillna(0).astype(int)
    return df


def main(path_to_fasta, metadata_file):
    seq = {}
    for genome in glob(f'{path_to_fasta}/*fna.gz'):
        print(genome)
        seq[genome] = extract_sequence_from_gzipped_fasta(genome)
    kmer_df = create_kmer_dataframe(seq, 8)

    return kmer_df
