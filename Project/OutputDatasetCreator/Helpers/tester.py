import pandas as pd

class Tester():
    txt_df = pd.DataFrame(columns=['file_idx','text','DD_Formatted'])
    rel_df = pd.DataFrame(columns=['file_idx','relationship_id','category','entity1','entity2','entity1_text','entity2_text','entity1_entity2','count_in_document'])
    ent_df = pd.DataFrame(columns=['file_idx','entity_id','category','start_idx','end_idx','text','orig_txt','count_in_document'])
    output_df = pd.DataFrame(columns=['file_idx','primary_diagnosis_NER','count', 'primary_diagnosis_LLM', 'Common_Underlying_Factors', 'primary_diagnosis', 'confidence'])

    def __init__(self, txt_df: pd.DataFrame, ent_df: pd.DataFrame, rel_df: pd.DataFrame, output_df: pd.DataFrame) -> None:
        self.txt_df = txt_df
        self.rel_df = rel_df
        self.ent_df = ent_df
        self.output_df = output_df

    def test_primary_medical_diagnosis(self):
        print("Testing...")
        count_matches = 0
        count_NER_matches = 0
        count_LLM_matches = 0
        for _, row in self.output_df.iterrows():
            if str(self.txt_df[self.txt_df['file_idx'] == row['file_idx']]['DD_Formatted'].iloc[0]) == str(row['primary_diagnosis']):
                count_matches += 1
            if str(self.txt_df[self.txt_df['file_idx'] == row['file_idx']]['DD_Formatted'].iloc[0]) == str(row['primary_diagnosis_NER']):
                count_NER_matches += 1
            if str(self.txt_df[self.txt_df['file_idx'] == row['file_idx']]['DD_Formatted'].iloc[0]) == str(row['primary_diagnosis_LLM']):
                count_LLM_matches += 1
        
        print("NER Accuracy: ", count_NER_matches/len(self.output_df))
        print("LLM Accuracy: ", count_LLM_matches/len(self.output_df))
        print("Ensemble Accuracy: ", count_matches/len(self.output_df))
    

    def test_common_underlying_factors(self):
        print("Testing...")
        count_matches = 0
        count_NER_matches = 0
        count_LLM_matches = 0
        for _, row in self.output_df.iterrows():
            if str(self.txt_df[self.txt_df['file_idx'] == row['file_idx']]['DD_Formatted'].iloc[0]) == str(row['primary_diagnosis']):
                count_matches += 1
            if str(self.txt_df[self.txt_df['file_idx'] == row['file_idx']]['DD_Formatted'].iloc[0]) == str(row['primary_diagnosis_NER']):
                count_NER_matches += 1
            if str(self.txt_df[self.txt_df['file_idx'] == row['file_idx']]['DD_Formatted'].iloc[0]) == str(row['primary_diagnosis_LLM']):
                count_LLM_matches += 1
        
        print("NER Accuracy: ", count_NER_matches/len(self.output_df))
        print("LLM Accuracy: ", count_LLM_matches/len(self.output_df))
        print("Ensemble Accuracy: ", count_matches/len(self.output_df))

        
        