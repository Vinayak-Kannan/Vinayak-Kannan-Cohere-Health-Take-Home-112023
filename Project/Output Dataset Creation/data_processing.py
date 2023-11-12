from typing import List, Dict
import pandas


class DataLoader():
    txt_df = None
    ent_df = None
    rel_df = None

    def __init__(self, txt_df: pandas.DataFrame, ent_df: pandas.DataFrame, rel_df: pandas.DataFrame) -> None:
        self.txt_df = txt_df
        self.ent_df = ent_df
        self.rel_df = rel_df
        pass

    def __summarize_discharge_diagnosis__(self):
        client = OpenAI(
            api_key='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',
        )
        messages_template = [
            {"role": "system", "content": "You will ingest a docuement and look for the 'Discharge Diagnosis' section. You will then extract each diagnosis from that section and return them seperated by a newline character. Do not change or alter the original diagnosis text. Remove any leading or trailing punctuation from the diagnosis."},
            {"role": "user", "content": "Who won the world series in 2020?"},
        ]
        self.txt_df['DD_Formatted'] = ""

        # Loop through each file in txt_df and get the 'Discharge Diagnosis' section
        for i, file_idx in enumerate(self.txt_df['file_idx'].unique()):
            messages_template[1]['content'] = self.txt_df[self.txt_df['file_idx'] == file_idx]['text'].iloc[0]
            while True:
                try:
                    response = client.chat.completions.create(
                        model="gpt-3.5-turbo-1106",
                        messages=messages_template, # type: ignore
                        temperature=0,
                        timeout=30,
                    )
                    if response.choices[0].message.content:
                        break
                    else:
                        print("Trying again")
                        time.sleep(1)
                except:
                    print("Trying again")
                    time.sleep(1)
            
            messages_template[1]['content'] = ""
            # Set DD_Formatted to the response from OpenAI
            self.txt_df.loc[self.txt_df['file_idx'] == file_idx, 'DD_Formatted'] = response.choices[0].message.content


    def __clean_data__(self, ) -> List[pandas.DataFrame]::
        """
        1. Convert 'text' columns to lowercase, in order to facilitate comparison.
        """
        self.ent_df['text'] = self.ent_df['text'].str.lower()
        self.ent_df['text'] = self.ent_df['text'].str.strip()

        """
        2. Remove \n ending from 'text' column in ent_df and in 'entity2' column in rel_df.
        """
        self.ent_df['text'] = self.ent_df['text'].str.rstrip('\n')
        self.rel_df['entity2'] = self.rel_df['entity2'].str.rstrip('\n')

        """
        3. Lemmatize entity texts and preserve original values in a new column
        """
        self.ent_df['orig_txt'] = self.ent_df['text']
        def lemmatize_text(text):
            words = nltk.word_tokenize(text)
            lemmatized_words = [lemmatizer.lemmatize(word) for word in words]
            return ' '.join(lemmatized_words)

        self.ent_df['text'] = self.ent_df['text'].apply(lemmatize_text)
        self.ent_df['text'] = self.ent_df['text'].astype(str)

        """
        4. Extract Dischrage Diagnosis using OpenAI LLM
        """
        self.__summarize_discharge_diagnosis__()

    def __feature_engineer__(self, rel_df: pandas.DataFrame, ent_df, txt_df) -> List[pandas.DataFrame]:
        """
        1. Join the appropriate entity1 and entity2 for each relation in rel_df.
        """
        # Remove first 5 letters (Arg Numbers) from 'entity1' and 'entity2' column in rel_df
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

        return [rel_df, ent_df, txt_df]