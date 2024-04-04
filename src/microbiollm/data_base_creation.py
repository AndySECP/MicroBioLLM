#! /usr/bin/env python3

import pandas as pd
from ete3 import NCBITaxa

"""
This script will generate a list of genomes to download from NCBI refseq from select superkingdom, for example Bacteria 
and a list of selected families from this super kingdom.
Example: ['Enterobacteriaceae', 'Bacillaceae', 'Pseudomonadaceae', 'Staphylococcaceae', 'Streptococcaceae']
"""

def tax_id_translator(tax_id: str):
    ncbi = NCBITaxa()
    name = ncbi.get_taxid_translator([tax_id])
    return name[tax_id]


def get_taxonomy(tax_id: str, rank: str):
    ncbi = NCBITaxa()
    try:
        lineage = ncbi.get_lineage(tax_id)
        rank_id = [key for key, value in ncbi.get_rank(lineage).items() if value == rank]
        return tax_id_translator(rank_id[0])
    except:
        return "NA"


def select_bacterial_set(families: list, bacteria_df):
    selected_genomes = pd.DataFrame()

    for family in families:
        genomes_family = bacteria_df[bacteria_df['family'] == family].sample(n=100, random_state=42)
        selected_genomes = pd.concat([selected_genomes, genomes_family])

    return selected_genomes


def save_list_to_file(list_to_save: list, filename: str):
    with open(filename, 'w') as f:
        for item in list_to_save:
            f.write(str(item) + '\n')


def main(assembly_summary_path: str, families: list, superkingdom: str = "Bacteria"):
    assembly_summary = pd.read_csv(assembly_summary_path, skiprows=1, sep="\t")
    complete_genome = assembly_summary[assembly_summary["assembly_level"] == "Complete Genome"]
    complete_genome['superkingdom'] = complete_genome['taxid'].apply(lambda x: get_taxonomy(str(x), "superkingdom"))
    filtered_genomes = complete_genome[complete_genome["superkingdom"] == superkingdom]
    filtered_genomes['family'] = complete_genome['taxid'].apply(lambda x: get_taxonomy(str(x), "family"))

    to_download = []
    selected_genomes = select_bacterial_set(families, filtered_genomes)

    for species in list(selected_genomes['ftp_path']):
        to_download.append(f'{species}/{species.split("/")[-1]}_genomic.fna.gz')

if __name__ == "__main__":
    main()

    # mettre argparse