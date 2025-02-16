# API KEY FOR FOOD DATA CENTRAL: wt36zfykN0hYir8znnddbb58xzj4PjOJYTZgXaOm

import pandas as pd
import ast

# ---------------------------
# (1) Processing data here
# REDO THIS IT'S WAY TOO SLOW ALSO WE ARE NOT ACTUALLY KEEPING THE DATASET IN GIT
# ---------------------------

def clean_ner(ner_str):
    try:
        ner_list = ast.literal_eval(ner_str)
    except Exception:
        ner_list = []
    return [ingredient.lower().strip() for ingredient in ner_list]

def clean_data(file_path: str, chunksize: int, processed_chunks: list):
    total = 0

    for chunk in pd.read_csv(file_path, chunksize = chunksize):

        chunk = chunk.drop(columns = ['link', 'source'], errors = 'ignore')

        chunk = chunk.dropna()

        chunk['title'] = chunk['title'].str.lower().str.strip()

        chunk['directions'] = chunk['directions'].str.lower().str.strip()

        chunk['clean_NER'] = chunk['NER'].apply(clean_ner)

        processed_chunks.append(chunk)

        total += chunksize

        print(f"Processed {total:02d} rows out of 2.23 million.")
    
    return pd.concat(processed_chunks, ignore_index = True)

chunksize = 10000
processed_chunks = []
file_path = 'full_dataset.csv'

processed_df = clean_data(file_path, chunksize, processed_chunks)

# ---------------------------
# (2) Only here to confirm content of DataFrame
# ---------------------------

pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', None) 

print(processed_df.head().to_string())
print(processed_df.shape)