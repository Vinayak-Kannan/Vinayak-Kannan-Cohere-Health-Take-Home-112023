import nltk
from load_data import load_ann, load_txt
import pandas as pd
from thefuzz import fuzz
from nltk.stem import WordNetLemmatizer
import spacy
from scispacy.abbreviation import AbbreviationDetector
from openai import OpenAI
from scipy.spatial.distance import cosine
from scispacy.linking import EntityLinker

nltk.download('punkt')
nltk.download('wordnet')

print("Running")

lemmatizer = WordNetLemmatizer()

PATH_TO_ZIP = "/workspaces/codespaces-jupyter/Project/RawData"
DATA_PATH = f"{PATH_TO_ZIP}/"
print(f"Full data path: {DATA_PATH}")
# read in txt files
txt_df = load_txt(DATA_PATH)
# read in REASONS entities from .ann files
ent_df, rel_df = load_ann(DATA_PATH)

"""
DATA CLEANING
"""

"""
1. Convert 'text' columns to lowercase, in order to facilitate comparison.
"""
# To lowercase 'text' column in ent_df
ent_df['text'] = ent_df['text'].str.lower()
ent_df['text'] = ent_df['text'].str.strip()

# To lowercase 'text' column in txt_df
# txt_df['text'] = txt_df['text'].str.lower()

"""
2. Remove \n ending from 'text' column in ent_df and in 'entity2' column in rel_df.
"""
ent_df['text'] = ent_df['text'].str.rstrip('\n')
rel_df['entity2'] = rel_df['entity2'].str.rstrip('\n')

"""
3. Convert 'start_idx' and 'end_idx' columns in ent_df to int.
"""
# Drop rows that cannot be converted to int TODO: Make this better
ent_df = ent_df[ent_df['start_idx'].str.isnumeric()]
ent_df = ent_df[ent_df['end_idx'].str.isnumeric()]
ent_df['start_idx'] = ent_df['start_idx'].astype(int)
ent_df['end_idx'] = ent_df['end_idx'].astype(int)

# Make new column with lemmatized text of 'text' column called 'lemmatized_text'
ent_df['orig_txt'] = ent_df['text']
def lemmatize_text(text):
    words = nltk.word_tokenize(text)
    lemmatized_words = [lemmatizer.lemmatize(word) for word in words]
    return ' '.join(lemmatized_words)

ent_df['text'] = ent_df['text'].apply(lemmatize_text)
# Convert ent_df['text'] to string
ent_df['text'] = ent_df['text'].astype(str)

"""
FEATURE ENGINEERING
"""

"""
1. Join the appropriate entity1 and entity2 for each relation in rel_df.
"""
# Remove first 5 letters from 'entity1' and 'entity2' column in rel_df
rel_df['entity1'] = rel_df['entity1'].str[5:]
rel_df['entity2'] = rel_df['entity2'].str[5:]
rel_df = rel_df.merge(ent_df[['entity_id', 'text', 'file_idx']], how='left', left_on=['entity1', 'file_idx'], right_on=['entity_id', 'file_idx'])
rel_df.rename(columns={'text': 'entity1_text'}, inplace=True)
rel_df = rel_df.merge(ent_df[['entity_id', 'text', 'file_idx']], how='left', left_on=['entity2', 'file_idx'], right_on=['entity_id', 'file_idx'])
rel_df.rename(columns={'text': 'entity2_text'}, inplace=True)
rel_df.drop(columns=['entity_id_x', 'entity_id_y'], inplace=True)

# Create column 'entity1_entity2' in rel_df
rel_df['entity1_entity2'] = rel_df['entity1_text'] + rel_df['entity2_text']

"""
2. Get count of text in file_idx for each entity in ent_df. Do the same for the 'entity1_entity2' in rel_df.
"""
ent_df_count = ent_df.groupby(['text', 'file_idx']).size().reset_index(name='count_in_document')
ent_df = ent_df.merge(ent_df_count, how='left', left_on=['text', 'file_idx'], right_on=['text', 'file_idx'])
rel_df_count = rel_df.groupby(['entity1_entity2', 'file_idx']).size().reset_index(name='count_in_document')
rel_df = rel_df.merge(rel_df_count, how='left', left_on=['entity1_entity2', 'file_idx'], right_on=['entity1_entity2', 'file_idx'])

"""
3. Create encoding to represent if entity in ent_df is in the 'Discharge Diagnosis', 'Chief Complaint', or 'History of Present Illness' section of the txt_df.
"""
def find_section_range(row, section_name):
    lines = row['text'].split('\n')
    matches = [(i, fuzz.ratio(line.lower(), section_name.lower())) for i, line in enumerate(lines)]
    matches.sort(key=lambda x: x[1], reverse=True)  # sort by fuzz.ratio in descending order
    if not matches:
        # Raise error if no match is found
        raise ValueError(f"Could not find section {section_name} in file {row['file_idx']}")
    start_line = matches[0][0]  # start of the range is the line with the highest fuzz.ratio
    end_line = start_line
    while end_line < len(lines) and lines[end_line].strip() != '':
        end_line += 1
    # calculate start and end index within the raw text
    start_index = sum(len(line) + 1 for line in lines[:start_line])  # +1 for the newline character
    end_index = sum(len(line) + 1 for line in lines[:end_line])  # +1 for the newline character
    return (start_index, end_index)

txt_df['DD_Range'] = txt_df.apply(lambda row: find_section_range(row, 'Discharge Diagnosis'), axis=1)
txt_df['CC_Range'] = txt_df.apply(lambda row: find_section_range(row, 'Chief Complaint'), axis=1)
txt_df['HPI_Range'] = txt_df.apply(lambda row: find_section_range(row, 'History of Present Illness'), axis=1)

# Join the 'DD_Range', 'CC_Range', and 'HPI_Range' columns from txt_df to ent_df
ent_df = ent_df.merge(txt_df[['file_idx', 'DD_Range', 'CC_Range', 'HPI_Range']], how='left', left_on=['file_idx'], right_on=['file_idx'])

# Loop through each entity in ent_df and check if it is in the 'Discharge Diagnosis', 'Chief Complaint', or 'History of Present Illness' section of the txt_df.
# If it is, then add the section name to the 'section' column in ent_df.
def find_section(row):
    # Throw error if start_idx is greater than end_idx
    # if row['start_idx'] > row['end_idx']:
    #     raise ValueError(f"start_idx {row['start_idx']} is greater than end_idx {row['end_idx']}")
    # # Throw error if start_idx and end_idx are in multiple sections
    # if row['start_idx'] >= row['DD_Range'][0] and row['end_idx'] <= row['DD_Range'][1] and row['start_idx'] >= row['CC_Range'][0] and row['end_idx'] <= row['CC_Range'][1]:
    #     print(row['file_idx'])
    #     # raise ValueError(f"start_idx {row['start_idx']} and end_idx {row['end_idx']} are in both DD and CC")
    # if row['start_idx'] >= row['DD_Range'][0] and row['end_idx'] <= row['DD_Range'][1] and row['start_idx'] >= row['HPI_Range'][0] and row['end_idx'] <= row['HPI_Range'][1]:
    #     print(row['file_idx'])
    #     # raise ValueError(f"start_idx {row['start_idx']} and end_idx {row['end_idx']} are in both DD and HPI")
    # if row['start_idx'] >= row['CC_Range'][0] and row['end_idx'] <= row['CC_Range'][1] and row['start_idx'] >= row['HPI_Range'][0] and row['end_idx'] <= row['HPI_Range'][1]:
    #     print(row['file_idx'])
        # raise ValueError(f"start_idx {row['start_idx']} and end_idx {row['end_idx']} are in both CC and HPI")
    
    # If start_idx and end_idx are in one section, return the section name
    if row['start_idx'] >= row['DD_Range'][0] and row['end_idx'] <= row['DD_Range'][1]:
        return 'Discharge Diagnosis'
    elif row['start_idx'] >= row['CC_Range'][0] and row['end_idx'] <= row['CC_Range'][1]:
        return 'Chief Complaint'
    elif row['start_idx'] >= row['HPI_Range'][0] and row['end_idx'] <= row['HPI_Range'][1]:
        return 'History of Present Illness'
    else:
        return 'Other'

ent_df['section'] = ent_df.apply(lambda row: find_section(row), axis=1)
# Drop DD_Range, CC_Range, and HPI_Range columns from ent_df
ent_df.drop(columns=['DD_Range', 'CC_Range', 'HPI_Range'], inplace=True)
# Apply one hot encoding to 'section' column in ent_df
ent_df = pd.get_dummies(ent_df, columns=['section'])

print("Running long part...")

nlp = spacy.load("en_core_sci_sm")
nlp.add_pipe("scispacy_linker", config={"resolve_abbreviations": True, "linker_name": "umls"})
linker = nlp.get_pipe("scispacy_linker")

df = pd.DataFrame(columns=['file_idx', 'primary_diagnosis', 'count'])  # initialize df as an empty DataFrame
def create_freq_dict(df_input):
    for i, file_idx in enumerate(txt_df['file_idx'].unique()):
        print(i, " out of ", len(txt_df['file_idx'].unique()))
        txt_df_subset = txt_df[txt_df['file_idx'] == file_idx]

        # Get DD_Range, CC_Range, and HPI_Range for each file in txt_df_subset and store in tuples
        DD_Range = txt_df_subset['DD_Range'].iloc[0]
        CC_Range = txt_df_subset['CC_Range'].iloc[0]
        HPI_Range = txt_df_subset['HPI_Range'].iloc[0]

        # Create a dict that loops through each line in the txt_df_subset 'text' column in the DD_Range and stores each diagnosis as a key and the value as an embedding of the diagnosis using get_embedding from openai.
        diagnosis_dict = {}
        for i, line in enumerate(str(txt_df_subset['text'].iloc[0][DD_Range[0]:DD_Range[1]]).split('\n')):
            if i == 0:
                continue
            if len(line) > 0:
                doc = nlp(line)
                for entity in doc.ents:
                    diagnosis_dict[entity] = []
                    for umls_ent in entity._.kb_ents:
                        try:
                            object = linker.kb.cui_to_entity[umls_ent[0]]
                            for alias in object[2]:
                                diagnosis_dict[entity].append(alias)
                        except Exception as e:
                            print(e)

        count_dict = {}
        for key, _ in diagnosis_dict.items():
            count_dict[key] = 0

        curr_text = str(txt_df_subset['text'].iloc[0])
        # text_to_analyze = ""
        # # Add the text from the CC_Range and HPI_Range to text_to_analyze
        # text_to_analyze += curr_text[CC_Range[0]:CC_Range[1]]
        # text_to_analyze += curr_text[HPI_Range[0]:HPI_Range[1]]

        for line in nltk.sent_tokenize(curr_text):
            doc = nlp(line)
            for entity in doc.ents:
                for umls_ent in entity._.kb_ents:
                    try:
                        object = linker.kb.cui_to_entity[umls_ent[0]]
                        for key, arr in diagnosis_dict.items():
                            for alias in object[2]:
                                if alias in arr:
                                    count_dict[key] += 1
                                    break
                    except Exception as e:
                        print(e)

        for key, value in count_dict.items():
            df_input = df_input.append({'file_idx': file_idx, 'primary_diagnosis': key, 'count': value}, ignore_index=True)

    return df_input

df = create_freq_dict(df)

df.to_csv('primary_diagnosis.csv', index=False)