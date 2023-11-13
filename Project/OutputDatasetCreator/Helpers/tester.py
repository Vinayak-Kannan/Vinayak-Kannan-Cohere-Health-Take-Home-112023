import pandas as pd
import numpy as np

class Tester():
    """Performs test exeuction to evalute performance of the model."""


    txt_df = pd.DataFrame(columns=['file_idx','text','DD_Formatted'])
    rel_df = pd.DataFrame(columns=['file_idx','relationship_id','category','entity1','entity2','entity1_text','entity2_text','entity1_entity2','count_in_document'])
    ent_df = pd.DataFrame(columns=['file_idx','entity_id','category','start_idx','end_idx','text','orig_txt','count_in_document'])
    output_df = pd.DataFrame(columns=['file_idx','primary_diagnosis_NER','count', 'primary_diagnosis_LLM', 'Common_Underlying_Factors', 'primary_diagnosis', 'confidence'])

    def __init__(self, txt_df: pd.DataFrame, ent_df: pd.DataFrame, rel_df: pd.DataFrame, output_df: pd.DataFrame) -> None:
        self.txt_df = txt_df
        self.rel_df = rel_df
        self.ent_df = ent_df
        self.output_df = output_df
    
    def test_primary_medical_diagnosis(self) -> None:
        """
        Runs evaluation metrics on the primary medical diagnosis identified by the model.
        """
        print("Testing Primary Medical diagnoses...")
        count_matches = 0
        count_NER_matches = 0
        count_LLM_matches = 0
        for _, row in self.output_df.iterrows():
            DD_rows = self.txt_df[self.txt_df['file_idx'] == row['file_idx']]['DD_Formatted'].iloc[0]
            first_DD = DD_rows.split('\n')[0]
            first_DD = first_DD.strip()
            first_DD = first_DD.lower()

            primary_diag = str(row['primary_diagnosis'])
            primary_diag = primary_diag.strip()
            primary_diag = primary_diag.lower()

            llm_diag = str(row['primary_diagnosis_LLM'])
            llm_diag = llm_diag.strip()
            llm_diag = llm_diag.lower()

            ner_diag = str(row['primary_diagnosis_NER'])
            ner_diag = ner_diag.strip()
            ner_diag = ner_diag.lower()
        
            ner_match = False
            llm_match = False
            if ner_diag in first_DD or first_DD in ner_diag:
                count_NER_matches += 1
                ner_match = True
            if llm_diag in first_DD or first_DD in llm_diag:
                count_LLM_matches += 1
                llm_match = True
            if ner_match or llm_match:
                count_matches += 1
        
        print("NER Accuracy: ", count_NER_matches/len(self.output_df))
        print("LLM Accuracy: ", count_LLM_matches/len(self.output_df))
        print("Ensemble Accuracy: ", count_matches/len(self.output_df))
    

    def test_common_underlying_factors(self) -> None:
        """
        Runs evaluation metrics on the common underlying factors identified by the model.
        """

        print("Testing Common Underlying Factors...")
        count_of_matches_between_ent_df_and_common_underlying_factors = 0

        for _, row in self.output_df.iterrows():
            file_idx = row['file_idx']
            # Get the Common_Underlying_Factors from the output_df
            common_underlying_factors = str(row['Common_Underlying_Factors'])
            common_underlying_factors = common_underlying_factors.strip()
            common_underlying_factors = common_underlying_factors.lower()
            common_underlying_factors = common_underlying_factors.split(',')
            # Trim to only the top 10
            common_underlying_factors = common_underlying_factors[:10]
            ent_df = self.ent_df[self.ent_df['file_idx'] == file_idx]
            ent_df = ent_df[ent_df['category'] == 'Reason']
            for factor in common_underlying_factors:
                factor = factor.lower()
                factor = factor.strip()
                if factor in ent_df['text'].values:
                    count_of_matches_between_ent_df_and_common_underlying_factors += 1
                    break
        
        print("Percentage of Notes with Underlying Factors that Directly Appear in Note", count_of_matches_between_ent_df_and_common_underlying_factors/len(self.output_df['file_idx'].unique()))
            