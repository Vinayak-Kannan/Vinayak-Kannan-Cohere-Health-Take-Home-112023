import time
from httpx import options
import spacy
import pandas as pd
import nltk
import scispacy
from scispacy.abbreviation import AbbreviationDetector
from scispacy.linking import EntityLinker
from openai import OpenAI


class PrimaryDiagnosisIdentifier:
    txt_df = pd.DataFrame()
    output_df_NER = pd.DataFrame(columns=["file_idx", "primary_diagnosis_NER", "count"])
    output_df_LLM = pd.DataFrame(columns=["file_idx", "primary_diagnosis_LLM"])
    output_df = pd.DataFrame(columns=["file_idx", "primary_diagnosis_NER", "count", "primary_diagnosis_LLM"])
    output_df_filtered = pd.DataFrame(columns=["file_idx", "primary_diagnosis", "count", "selected_approach", "confidence"])

    def __init__(self, txt_df: pd.DataFrame) -> None:
        self.txt_df = txt_df

    def process_data(self) -> pd.DataFrame:
        print("Processing Data...")
        self.output_df_NER = self.__create_primary_diagnosis_NER(self.txt_df)
        self.output_df_NER = self.output_df_NER.sort_values(by=['file_idx', 'count', 'primary_diagnosis_NER'], ascending=[False, False, True]).drop_duplicates(subset=['file_idx'], keep='first')

        self.output_df_LLM = self.__create_primary_diagnosis_LLM(self.txt_df)
        self.output_df_LLM = self.output_df_LLM.sort_values(by=['file_idx'], ascending=[False])

        # Merge output_df_LLM and output_df_NER
        self.output_df = self.output_df_NER.merge(self.output_df_LLM, on='file_idx')
        self.output_df_filtered = self.__select_approach(self.output_df)

        return self.output_df_filtered

    def __select_approach(self, output_df: pd.DataFrame) -> pd.DataFrame:
        output_df['selected_approach'] = ''
        output_df['confidence'] = ''
        output_df['primary_diagnosis'] = ''
        for index, row in output_df.iterrows():
            if row['primary_diagnosis_NER'] == row['primary_diagnosis_LLM']:
                output_df.at[index, 'selected_approach'] = 'BOTH'
                output_df.at[index, 'confidence'] = 'Higher Confidence Prediction'
                output_df.at[index, 'primary_diagnosis'] = str(row['primary_diagnosis_NER'])
            elif 'discharge diagnosis' in str(self.txt_df[self.txt_df['file_idx'] == row['file_idx']].iloc[0]).lower() or 'discharge diagnoses' in str(self.txt_df[self.txt_df['file_idx'] == row['file_idx']].iloc[0]).lower():
                output_df.at[index, 'selected_approach'] = 'NER'
                output_df.at[index, 'confidence'] = 'Lower Confidence Prediction'
                output_df.at[index, 'primary_diagnosis'] = str(row['primary_diagnosis_NER'])
            else:
                output_df.at[index, 'selected_approach'] = 'LLM'
                output_df.at[index, 'confidence'] = 'Lower Confidence Prediction'
                output_df.at[index, 'primary_diagnosis'] = str(row['primary_diagnosis_LLM'])
        output_df = output_df.drop(columns=['primary_diagnosis_NER', 'primary_diagnosis_LLM'])
        return output_df


    def __create_primary_diagnosis_NER(self, txt_df: pd.DataFrame) -> pd.DataFrame:
        nltk.download('punkt')
        nltk.download('wordnet')
        nlp = spacy.load("en_core_sci_sm")
        nlp.add_pipe("abbreviation_detector")
        nlp.add_pipe(
            "scispacy_linker",
            config={"resolve_abbreviations": True, "linker_name": "umls"},
        )
        linker = nlp.get_pipe("scispacy_linker")

        local_output_df = pd.DataFrame(
            columns=["file_idx", "primary_diagnosis", "count"]
        )

        for i, file_idx in enumerate(txt_df["file_idx"].unique()):
            print(i, " out of ", len(txt_df["file_idx"].unique()), " files processed")
            txt_df_subset = txt_df[txt_df["file_idx"] == file_idx]

            diagnosis_dict = {}
            for i, line in enumerate(
                str(txt_df_subset["DD_Formatted"].iloc[0]).split("\n")
            ):
                if "discharge diagnosis" in line.lower() or "discharge diagnoses" in line.lower():
                    continue
                if len(line) > 0:
                    doc = nlp(line)
                    for entity in doc.ents:
                        diagnosis_dict[line] = []
                        for umls_ent in entity._.kb_ents:
                            try:
                                object = linker.kb.cui_to_entity[umls_ent[0]]
                                for alias in object[2]:
                                    diagnosis_dict[line].append(alias)
                            except Exception as e:
                                print(e)

            count_dict = {}
            for key, _ in diagnosis_dict.items():
                count_dict[key] = 0

            curr_text = str(txt_df_subset["text"].iloc[0])

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
                key = key.strip().title()
                while not key[0].isalpha():
                    key = key[1:]
                local_output_df = local_output_df.append({"file_idx": file_idx, "primary_diagnosis_NER": str(key), "count": value}, ignore_index=True)  # type: ignore

        return local_output_df

    def __create_primary_diagnosis_LLM(self, txt_df: pd.DataFrame) -> pd.DataFrame:
        client = OpenAI(
            api_key="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",
        )
        messages_template = [
            {
                "role": "system",
                "content": """You will ingest a docuement about a clinicians note detailing a doctor visit. 
                I will give you a list of possible Primary Medical Diagnoses; each is seperated by a ','. You will review the docuement and select the most likely Primary Medical Diagnoses from the list I provide.
                A 'Primary Medical Diagnosis' is defined as the 'Main condition treated or investigated during relevant episode of healthcare'.
                You will return the option that best respresents the Primary Medical Diagnoses from the list I've provided.
                Do not change or alter the original diagnosis text from the list I provided.
                Remove any leading or trailing punctuation from the diagnosis.""",
            },
            {
                "role": "user",
                "content": ""
            },
            {
                "role": "user",
                "content": ""
            },
        ]

        local_output_df = pd.DataFrame(
            columns=["file_idx", "primary_diagnosis"]
        )

        # Loop through each file in txt_df and get the 'text'
        for _, file_idx in enumerate(txt_df["file_idx"].unique()):
            options = "These are the list of possible Primary Medical Diagnoses: \n\n"
            raw_text = "This is the raw text from the clinicians note: \n\n"
            messages_template[2]["content"] = raw_text + txt_df[txt_df["file_idx"] == file_idx][
                "text"
            ].iloc[0]
            for _, line in enumerate(
                str(txt_df[txt_df["file_idx"] == file_idx].iloc[0]).split("\n")
            ):
                if "discharge diagnosis" in line.lower() or "discharge diagnoses" in line.lower():
                    continue
                if len(line) > 0:
                    options += line + ","
            messages_template[1]["content"] = options

            while True:
                try:
                    response = client.chat.completions.create(
                        model="gpt-3.5-turbo-1106",
                        messages=messages_template,  # type: ignore
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

            response = response.choices[0].message.content.split('"')[1]
            local_output_df = local_output_df.append({"file_idx": file_idx, "primary_diagnosis_LLM": response}, ignore_index=True) # type: ignore
        
        return local_output_df