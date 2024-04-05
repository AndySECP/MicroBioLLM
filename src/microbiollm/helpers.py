import subprocess

from ete3 import NCBITaxa


def execute_command(command: str):
    try:
        # Execute the command
        subprocess.run(command.split(), check=True)
        print("Command executed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error executing command: {e}")


def tax_id_translator(tax_id: str) -> str:
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
