import requests
import io

import numpy as np
import pandas as pd

import seaborn as sns
from matplotlib import pyplot as plt

from Bio import SeqIO

taxon_df = pd.read_csv("../progen_esm_compare/uniprot_top_species_with_taxon.csv")
taxon_ids = list(taxon_df.taxon_id_final.values)
species = list(taxon_df.Species.values)

url_start = f'https://rest.uniprot.org/uniprotkb/stream?compressed=false&format=tsv&query=(reviewed:true)%20AND%20'

df_list = []
for taxon_id, specie in zip(taxon_ids, species):
    url_end = f'(taxonomy_id:{taxon_id})'
    url = url_start + url_end
    all_tsvs = requests.get(url).text
    temp_df = pd.read_csv(io.StringIO(all_tsvs), sep="\t")
    temp_df['Gene_processed'] = [name.split('_')[0] for name in  temp_df['Entry Name'].values]
    temp_df['taxon_id'] = taxon_id
    temp_df['long_species'] = specie
    df_list.append(temp_df)

df = pd.concat(df_list)

# Get FASTA format sequences for each gene in the above df
fasta_url_start = f'https://rest.uniprot.org/uniprotkb/stream?compressed=false&format=fasta&query=(reviewed:true)%20AND%20'

def fasta_to_df_row(seqrecord):
    df_row = {}
    df_row['Entry'] = seqrecord.id.split("|")[1]
    df_row['Entry Name'] = seqrecord.id.split("|")[2]
    df_row['Description'] = seqrecord.description
    df_row['Sequence'] = str(seqrecord.seq)
    return df_row

list_of_rows = []
for taxon_id, specie in zip(taxon_ids, species):
    url_end = f'(taxonomy_id:{taxon_id})'
    url = fasta_url_start + url_end
    fasta_text = requests.get(url).text    
    fasta_io = io.StringIO(fasta_text)
    all_fastas = list(SeqIO.parse(fasta_io, 'fasta'))
    
    this_specie_rows = [fasta_to_df_row(seqrecord) for seqrecord in all_fastas]
    list_of_rows.append(this_specie_rows)

from itertools import chain
flattened_list = list(chain.from_iterable(list_of_rows))
seq_df = pd.DataFrame(flattened_list)
