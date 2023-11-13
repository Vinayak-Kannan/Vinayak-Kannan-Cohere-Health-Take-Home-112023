from openai import OpenAI
import pandas as pd
from Helpers.data_loader import DataLoader
from Helpers.underlying_factor_identifier import UnderlyingFactorIdentifier
from Helpers.primary_diagnosis_identifier import PrimaryDiagnosisIdentifier


class ClinicianNoteDataSetHandler():
    """Handles exeucting pipeline to create dataset with identified primary diagnosis and underlying factors"""

    txt_df = pd.DataFrame()
    rel_df = pd.DataFrame()
    ent_df = pd.DataFrame()
    OpenAI_API_KEY = ""


    def __init__(self, txt_df: pd.DataFrame, ent_df: pd.DataFrame, rel_df: pd.DataFrame, OpenAI_API_KEY: str) -> None:
        self.txt_df = txt_df
        self.rel_df = rel_df
        self.ent_df = ent_df
        self.OpenAI_API_KEY = OpenAI_API_KEY

    def identify_primary_diagnosis_and_underlying_factors(self) -> pd.DataFrame:
        """
        Runs raw text dataframes through data set generation pipeilne and returns data
        frame with primary diagnosis and underlying factors identified for each file
    
    
        Returns
        -------
        pandas DataFrame
            Df indexed by the file_idx field, containing primary diagnosis, underlying factors,
            confidence statements, and outputs for each model used in ensemble method.
        """
        txt_df = self.txt_df
        ent_df = self.ent_df
        rel_df = self.rel_df
        if len(self.OpenAI_API_KEY) == 0:
            "Please provide an OpenAI API Key and try again"
            return pd.DataFrame()

        data_loader = DataLoader(txt_df, ent_df, rel_df, self.OpenAI_API_KEY)
        data_loader.clean_data()
        txt_df, ent_df, rel_df = data_loader.get_data()

        txt_df.to_csv('./Intermediate Data File/txt_df.csv', index=False)
        ent_df.to_csv('./Intermediate Data File/ent_df.csv', index=False)
        rel_df.to_csv('./Intermediate Data File/rel_df.csv', index=False)

        primary_diagnosis_identifier = PrimaryDiagnosisIdentifier(txt_df, self.OpenAI_API_KEY)
        primary_diagnosis_df = primary_diagnosis_identifier.process_data()

        underlying_factor_identifier = UnderlyingFactorIdentifier(txt_df, ent_df, rel_df)
        underlying_factor_df = underlying_factor_identifier.process_dataset()

        merged_output_with_diagnosis_and_factors_df = primary_diagnosis_df.merge(underlying_factor_df, on='file_idx')
        merged_output_with_diagnosis_and_factors_df['file_idx'] = merged_output_with_diagnosis_and_factors_df['file_idx'].astype(int)
        merged_output_with_diagnosis_and_factors_df['count'] = merged_output_with_diagnosis_and_factors_df['count'].astype(int)
        merged_output_with_diagnosis_and_factors_df['primary_diagnosis'] = merged_output_with_diagnosis_and_factors_df['primary_diagnosis'].astype(str)
        merged_output_with_diagnosis_and_factors_df = merged_output_with_diagnosis_and_factors_df.sort_values(by=['file_idx', 'count', 'primary_diagnosis'], ascending=[False, False, True]).drop_duplicates(subset=['file_idx'], keep='first')

        return merged_output_with_diagnosis_and_factors_df