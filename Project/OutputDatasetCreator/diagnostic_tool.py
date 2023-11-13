from Helpers.primary_diagnosis_identifier import PrimaryDiagnosisIdentifier
from Helpers.data_loader import DataLoader

import pandas as pd

class DiagnosticTool():
    text = str
    ent_df = pd.DataFrame()
    rel_df = pd.DataFrame()


    def __init__(self, text, ent_df: pd.DataFrame, rel_df: pd.DataFrame) -> None:
        self.text = text
        self.ent_df = ent_df
        self.rel_df = rel_df

    def get_primary_diagnosis(self):
        # Create DF with one row with text. Columns are file_idx, text, DD_Formatted
        txt_df = pd.DataFrame(columns=['file_idx','text','DD_Formatted'])
        txt_df = txt_df.append({'file_idx': 0, 'text': self.text, 'DD_Formatted': ''}, ignore_index=True) # type: ignore
        DataLoaderTool = DataLoader(txt_df, self.ent_df, self.rel_df)
        txt_df = DataLoaderTool.summarize_discharge_diagnosis(txt_df)

        output_df_filtered = PrimaryDiagnosisIdentifier(txt_df).process_data()

        # If confidence is 'Higher Confidence Prediction', return primary_diagnosis
        if output_df_filtered['confidence'].iloc[0] == 'Higher Confidence Prediction':
            return (str(output_df_filtered['primary_diagnosis'].iloc[0]), "", str(output_df_filtered['confidence'].iloc[0]))
        else:
            return (str(output_df_filtered['primary_diagnosis_NER'].iloc[0]), str(output_df_filtered['primary_diagnosis_LLM'].iloc[0]), "")
    

