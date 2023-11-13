import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from Helpers.load_data import load_ann, load_txt
from Helpers.clinician_note_dataset_handler import ClinicianNoteDataSetHandler

# Provided code by Cohere Health
PATH_TO_ZIP = "/workspaces/codespaces-jupyter/Project/RawData"
DATA_PATH = f"{PATH_TO_ZIP}/"
print(f"Full data path: {DATA_PATH}")
txt_df = load_txt(DATA_PATH)
ent_df, rel_df = load_ann(DATA_PATH)

data_handler = ClinicianNoteDataSetHandler(txt_df, ent_df, rel_df)

output = data_handler.identify_primary_diagnosis_and_underlying_factors()
output.to_csv('./Clinical_Note_Diagnoses_Factors_Dataset.csv', index=False)