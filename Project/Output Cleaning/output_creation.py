import pandas as pd

# Read in primary_diagnosis.csv and top_5_reason_ent.csv
primary_diagnosis = pd.read_csv('primary_diagnosis.csv')
top_5_reason_ent = pd.read_csv('top_5_reason_ent.csv')

# Merge primary_diagnosis and top_5_reason_ent on 'file_idx'
primary_diagnosis = primary_diagnosis.merge(top_5_reason_ent, on='file_idx')

def process_common_factors(row):
    factors = row['Common_Underlying_Factors'].split(',')[:5]
    factors = [factor.strip().title() for factor in factors]
    return ','.join(factors)

primary_diagnosis['Common_Underlying_Factors'] = primary_diagnosis.apply(process_common_factors, axis=1)

# Save primary_diagnosis to a csv file
primary_diagnosis.to_csv('output_dataset.csv', index=False)
