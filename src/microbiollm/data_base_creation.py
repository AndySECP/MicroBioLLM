#! /usr/bin/env python3

import argparse
import logging
from pathlib import Path
import requests
import sys

import pandas as pd

from helpers import get_taxonomy

"""
This script will generate a list of genomes to download from NCBI (Refseq or Genbank) from select superkingdom, for example Bacteria 
and a list of selected families from this super kingdom.
Example: ['Enterobacteriaceae', 'Bacillaceae', 'Pseudomonadaceae', 'Staphylococcaceae', 'Streptococcaceae']
It will download all the assemblies and store them for the next steps of the pipeline. 
This is used for the first proof of concept so the database will be sample and way more simple compared to a real microbiome

Example use: 
python3 MicroBioLLM/src/microbiollm/data_base_creation.py -d refseq -k Bacteria -f Bacillaceae Pseudomonadaceae -s 10 -g database

"""

logging.basicConfig(level=logging.INFO,
                    format='[%(filename)s:%(lineno)s - %(funcName)s()] [%(asctime)s %(levelname)s] %(message)s')
baselog = logging.getLogger(__name__)
baselog.setLevel(logging.INFO)


class GenomeDataBase:
    def __init__(self, sample_size: int, families: list, superkingdom: str = "Bacteria", database: str = "refseq",
                 genome_dir: str = "database"):
        """
        This class will generate the database of genomes later used for training and testing the model
        :param sample_size: number of genomes to include in the database
        :param families: list of families to download
        :param superkingdom: e.g. Bacteria
        :param database: NCBI database to download from (refseq or genbank)
        :param genome_dir: directory to save the data in
        """
        logger = baselog.getChild('Create genome database')

        self.sample_size = sample_size
        self.database = database
        self.families = families
        self.superkingdom = superkingdom
        self.genome_dir = genome_dir
        if Path(f'{self.genome_dir}/assembly_summary_{self.database}.txt').exists():
            logger.info("Assembly file already exist - won't be downloaded")
        else:
            # download the assembly file from NCBI
            self.download_assembly_summary()

        self.assembly_file_df = pd.read_csv(f"{self.genome_dir}/assembly_summary_{self.database}.txt", skiprows=1, sep= "\t")
        self.filtered_assembly_file = self.filter_assembly_file()
        # save the selected organisms data
        self.filtered_assembly_file.to_csv(f"{self.genome_dir}/selected_organisms.csv")
        self.selected_orga_df = self.select_organisms()

    def download_assembly_summary(self):
        """
        Download the assembly file from NCBI
        """
        logger = baselog.getChild('Download assembly summary')
        logger.info(f"Start downloading assembly_summary_{self.database}.txt")

        assembly_url = f"https://ftp.ncbi.nlm.nih.gov/genomes/{self.database}/assembly_summary_{self.database}.txt"
        response = requests.get(assembly_url)
        directory = Path(self.genome_dir)
        directory.mkdir(parents=True, exist_ok=True)

        with open(f"{self.genome_dir}/assembly_summary_{self.database}.txt", 'wb') as f:
            f.write(response.content)
        logger.info(f"assembly_summary_{self.database}.txt has been downloaded successfully")

    def filter_assembly_file(self):
        """
        Filter the assembly file to keep only complete genome and the superkingdom to download
        """
        complete_genome = self.assembly_file_df[self.assembly_file_df["assembly_level"] == "Complete Genome"]
        complete_genome['superkingdom'] = complete_genome['taxid'].apply(lambda x: get_taxonomy(str(x), "superkingdom"))
        filtered_assembly_file = complete_genome[complete_genome["superkingdom"] == self.superkingdom]
        filtered_assembly_file['family'] = filtered_assembly_file['taxid'].apply(
            lambda x: get_taxonomy(str(x), "family"))

        return filtered_assembly_file

    def select_organisms(self):
        """
        Select a random sample of genomes from the selected families
        """
        selected_genomes = pd.DataFrame()

        for family in self.families:
            genomes_family = self.filtered_assembly_file[self.filtered_assembly_file['family'] == family].sample(n=self.sample_size,
                                                                                                                 random_state=42)
            selected_genomes = pd.concat([selected_genomes, genomes_family])

        return selected_genomes

    def download_genomes(self):
        """
        Download selected assemblies from NCBI
        """
        logger = baselog.getChild('Download genomes')
        for species in list(self.selected_orga_df['ftp_path']):
            species_url = f'{species}/{species.split("/")[-1]}_genomic.fna.gz'
            logger.info(f"Downloading {species_url}")
            response = requests.get(species_url)

            # Extract filename from URL and save to the current directory
            file_name = species_url.split('/')[-1]
            with open(f'{self.genome_dir }/{file_name}', 'wb') as f:
                f.write(response.content)


def create_data_base(args):
    genomes = GenomeDataBase(int(args.sample_size), args.families, args.superkingdom, args.database, args.genome_dir)
    # download the genomes:
    genomes.download_genomes()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create genome database')
    parser.add_argument('-d', '--database', dest='database', help='refseq or genbank', required=True)
    parser.add_argument('-k', '--superkingdom', dest='superkingdom', help='e.g: Bacteria', required=True)
    parser.add_argument('-f', '--families', dest='families', help='e.g Bacillaceae Pseudomonadaceae', nargs='+', required=True)
    parser.add_argument('-s', '--sample_size', dest='sample_size', help='number of organism to download', required=True)
    parser.add_argument('-g', '--genome_dir', dest='genome_dir', help='directory to store the genomes', required=True)

    args = parser.parse_args()

    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

    create_data_base(args)
