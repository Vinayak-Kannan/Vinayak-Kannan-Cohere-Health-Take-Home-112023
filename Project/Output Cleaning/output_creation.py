import pandas as pd

# Read in primary_diagnosis.csv and top_5_reason_ent.csv
primary_diagnosis = pd.read_csv('primary_diagnosis.csv')
top_5_reason_ent = pd.read_csv('top_5_reason_ent.csv')

# Merge primary_diagnosis and top_5_reason_ent on 'file_idx'
primary_diagnosis = primary_diagnosis.merge(top_5_reason_ent, on='file_idx')