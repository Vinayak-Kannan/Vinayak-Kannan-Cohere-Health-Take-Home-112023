import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from dotenv import dotenv_values
from Helpers.load_data import load_ann, load_txt
from Helpers.clinician_note_dataset_handler import ClinicianNoteDataSetHandler

# Provided code by Cohere Health
PATH_TO_ZIP = "/workspaces/codespaces-jupyter/Project/RawData"
DATA_PATH = f"{PATH_TO_ZIP}/"
print(f"Full data path: {DATA_PATH}")
txt_df = load_txt(DATA_PATH)
ent_df, rel_df = load_ann(DATA_PATH)

API_Key = ""
# Load API Key from .env file if API_KEY is not provided
config = dotenv_values("Project/.env")
API_Key = config.get("OPEN_AI_KEY")
if API_Key is None:
    print("Please provide an OpenAI API Key and try again")
    exit()
data_handler = ClinicianNoteDataSetHandler(txt_df, ent_df, rel_df, API_Key)

output = data_handler.identify_primary_diagnosis_and_underlying_factors()
output.to_csv('./Clinical_Note_Diagnoses_Factors_Dataset.csv', index=False)