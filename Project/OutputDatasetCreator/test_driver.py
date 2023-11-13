from Helpers.tester import Tester
import pandas as pd

output_df = pd.read_csv('./Clinical_Note_Diagnoses_Factors_Dataset.csv')
txt_df = pd.read_csv('./txt_df.csv')
ent_df= pd.read_csv('./ent_df.csv')
rel_df = pd.read_csv('./rel_df.csv')

Tester = Tester(txt_df, ent_df, rel_df, output_df)
Tester.test_primary_medical_diagnosis()

Tester.test_common_underlying_factors()