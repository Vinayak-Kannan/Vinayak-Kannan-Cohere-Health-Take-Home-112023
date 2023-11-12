import pandas as pd
from Helpers.data_loader import DataLoader
from Helpers.underlying_factor_identifier import UnderlyingFactorIdentifier
from Helpers.primary_diagnosis_identifier import PrimaryDiagnosisIdentifier


class ClinicianNoteDataSetHandler():
    txt_df = pd.DataFrame()
    rel_df = pd.DataFrame()
    ent_df = pd.DataFrame()


    def __init__(self, txt_df: pd.DataFrame, ent_df: pd.DataFrame, rel_df: pd.DataFrame) -> None:
        self.txt_df = txt_df
        self.rel_df = rel_df
        self.ent_df = ent_df

    def identify_primary_diagnosis_and_underlying_factors(self) -> pd.DataFrame:
        txt_df = self.txt_df
        ent_df = self.ent_df
        rel_df = self.rel_df

        data_loader = DataLoader(txt_df, ent_df, rel_df)
        data_loader.clean_data()
        txt_df, ent_df, rel_df = data_loader.get_data()

        primary_diagnosis_identifier = PrimaryDiagnosisIdentifier(txt_df)
        primary_diagnosis_df = primary_diagnosis_identifier.process_data()

        underlying_factor_identifier = UnderlyingFactorIdentifier(txt_df, ent_df, rel_df)
        underlying_factor_df = underlying_factor_identifier.process_dataset()

        merged_output_with_diagnosis_and_factors_df = primary_diagnosis_df.merge(underlying_factor_df, on='file_idx')

        return merged_output_with_diagnosis_and_factors_df


