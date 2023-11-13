import time
from typing import List
from openai import OpenAI
import pandas as pd
import nltk
from nltk.stem import WordNetLemmatizer


class DataLoader:
    txt_df = pd.DataFrame()
    ent_df = pd.DataFrame()
    rel_df = pd.DataFrame()
    lemmatizer = WordNetLemmatizer()

    def __init__(
        self, txt_df: pd.DataFrame, ent_df: pd.DataFrame, rel_df: pd.DataFrame
    ) -> None:
        self.txt_df = txt_df
        self.ent_df = ent_df
        self.rel_df = rel_df

    def clean_data(self) -> None:
        print("Cleaning Data...")
        self.rel_df, self.ent_df, self.txt_df = self.__clean_data(
            self.rel_df, self.ent_df, self.txt_df
        )
        print("Feature Engineering...")
        self.rel_df, self.ent_df, self.txt_df = self.__feature_engineer(
            self.rel_df, self.ent_df, self.txt_df
        )

    def get_data(self) -> List[pd.DataFrame]:
        return [self.txt_df, self.ent_df, self.rel_df]

    def summarize_discharge_diagnosis(self, txt_df: pd.DataFrame) -> pd.DataFrame:
        client = OpenAI(
            api_key="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",
        )
        messages_template = [
            {
                "role": "system",
                "content": "You will ingest a docuement and look for the 'Discharge Diagnosis' section. You will then extract each diagnosis from that section and return them seperated by a newline character. Do not change or alter the original diagnosis text. Remove any leading or trailing punctuation from the diagnosis.",
            },
            {"role": "user", "content": ""},
        ]
        txt_df["DD_Formatted"] = ""

        # Loop through each file in txt_df and get the 'Discharge Diagnosis' section
        for i, file_idx in enumerate(txt_df["file_idx"].unique()):
            print(str(i) + " out of " + str(len(txt_df["file_idx"].unique())))
            messages_template[1]["content"] = txt_df[txt_df["file_idx"] == file_idx][
                "text"
            ].iloc[0]
            while True:
                try:
                    response = client.chat.completions.create(
                        model="gpt-4-1106-preview",
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

            messages_template[1]["content"] = ""
            # Set DD_Formatted to the response from OpenAI
            txt_df.loc[
                txt_df["file_idx"] == file_idx, "DD_Formatted"
            ] = response.choices[0].message.content

        return txt_df

    def __clean_data(
        self, rel_df: pd.DataFrame, ent_df: pd.DataFrame, txt_df: pd.DataFrame
    ) -> List[pd.DataFrame]:
        """
        1. Convert 'text' columns to lowercase, in order to facilitate comparison.
        """
        ent_df["text"] = ent_df["text"].str.lower()
        ent_df["text"] = ent_df["text"].str.strip()

        """
        2. Remove \n ending from 'text' column in ent_df and in 'entity2' column in rel_df.
        """
        ent_df["text"] = ent_df["text"].str.rstrip("\n")
        rel_df["entity2"] = rel_df["entity2"].str.rstrip("\n")

        """
        3. Lemmatize entity texts and preserve original values in a new column
        """
        ent_df["orig_txt"] = ent_df["text"]

        def lemmatize_text(text):
            words = nltk.word_tokenize(text)
            lemmatized_words = [self.lemmatizer.lemmatize(word) for word in words]
            return " ".join(lemmatized_words)

        ent_df["text"] = ent_df["text"].apply(lemmatize_text)
        ent_df["text"] = ent_df["text"].astype(str)

        return [rel_df, ent_df, txt_df]

    def __feature_engineer(
        self, rel_df: pd.DataFrame, ent_df: pd.DataFrame, txt_df: pd.DataFrame
    ) -> List[pd.DataFrame]:
        """
        1. Join the appropriate entity1 and entity2 for each relation in rel_df.
        """
        # Remove first 5 letters (Arg Numbers) from 'entity1' and 'entity2' column in rel_df
        rel_df["entity1"] = rel_df["entity1"].str[5:]
        rel_df["entity2"] = rel_df["entity2"].str[5:]
        rel_df = rel_df.merge(
            ent_df[["entity_id", "text", "file_idx"]],
            how="left",
            left_on=["entity1", "file_idx"],
            right_on=["entity_id", "file_idx"],
        )
        rel_df.rename(columns={"text": "entity1_text"}, inplace=True)
        rel_df = rel_df.merge(
            ent_df[["entity_id", "text", "file_idx"]],
            how="left",
            left_on=["entity2", "file_idx"],
            right_on=["entity_id", "file_idx"],
        )
        rel_df.rename(columns={"text": "entity2_text"}, inplace=True)
        rel_df.drop(columns=["entity_id_x", "entity_id_y"], inplace=True)

        # Create column 'entity1_entity2' in rel_df
        rel_df["entity1_entity2"] = rel_df["entity1_text"] + ":" + rel_df["entity2_text"]

        """
        2. Get count of text in file_idx for each entity in ent_df. Do the same for the 'entity1_entity2' in rel_df.
        """
        ent_df_count = (
            ent_df.groupby(["text", "file_idx"])
            .size()
            .reset_index(name="count_in_document")
        )
        ent_df = ent_df.merge(
            ent_df_count,
            how="left",
            left_on=["text", "file_idx"],
            right_on=["text", "file_idx"],
        )
        rel_df_count = (
            rel_df.groupby(["entity1_entity2", "file_idx"])
            .size()
            .reset_index(name="count_in_document")
        )
        rel_df = rel_df.merge(
            rel_df_count,
            how="left",
            left_on=["entity1_entity2", "file_idx"],
            right_on=["entity1_entity2", "file_idx"],
        )

        """
        3. Extract Discharge Diagnosis using OpenAI LLM
        """
        txt_df = self.summarize_discharge_diagnosis(txt_df)

        
        return [rel_df, ent_df, txt_df]
