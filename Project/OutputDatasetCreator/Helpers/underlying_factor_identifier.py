import pandas as pd
from Bio_Epidemiology_NER.bio_recognizer import ner_prediction
from nltk.stem import WordNetLemmatizer


class UnderlyingFactorIdentifier:
    """Identifies underlying factors for each clinician note using NER across all documents"""


    ent_df = pd.DataFrame()
    rel_df = pd.DataFrame()
    txt_df = pd.DataFrame()
    lemmatizer = WordNetLemmatizer()

    def __init__(
        self, txt_df: pd.DataFrame, ent_df: pd.DataFrame, rel_df: pd.DataFrame
    ) -> None:
        self.txt_df = txt_df
        self.ent_df = ent_df
        self.rel_df = rel_df

    def process_dataset(self) -> pd.DataFrame:
        """
        Runs pipeline to identify underlying factors for each clinician note using NER across all documents

        Returns
        -------
        pandas DataFrame
            Dataframe containing identified underlying factors for each clinician note.
        """

        print("Processing Underlying Factors...")
        self.__connect_reasons_to_drugs(self.rel_df, self.ent_df)
        top_5_factors_by_file = self.__parse_top_5_factors()
        return top_5_factors_by_file

    def __connect_reasons_to_drugs(self, rel_df: pd.DataFrame, ent_df: pd.DataFrame):
        """
        Creates dataframe marrying 'Reason' and 'Drug' entities together, in order to identify underlying factors
        tied to each drug within a clinician note.
    
        Parameters
        ----------
        rel_df: pandas DataFrame
            Dataframe containing RE information between different entities across docuements.
        ent_df: pandas DataFrame
            Dataframe containing Named Entity informaiton for different entities across docuements.

        Returns
        -------
        pandas DataFrame
            Intermediary Dataframe containing 'Reason' and 'Drug' entities tied together.
        """

        print("Tying reasons to drugs to identify underlying factors...")
        # Make a dataframe with only 'Reason' category and 'Drug' category
        ent_df = ent_df[ent_df["category"].isin(["Reason", "Drug"])]
        ent_df["Joined_Reason"] = ""

        ent_df.loc[ent_df["category"] == "Reason", "Reason_From_Drug"] = ent_df["text"]

        # Filter rel_df to only 'Reason-Drug' category and sort by count across docuements
        rel_df = rel_df[rel_df["category"] == "Reason-Drug"].sort_values(
            by="count_in_document", ascending=False
        )
        rel_df_reason_drug_count = (
            rel_df.groupby(["entity1_entity2"])
            .agg({"count_in_document": "sum"})
            .reset_index()
        )

        # Create column in rel_df_reason_drug_count that has 'Reason' and 'Drug'. 'Reason' is the first part of the 'entity1_entity2' and 'Drug' is the second part of the 'entity1_entity2', seperated by a ':'
        rel_df_reason_drug_count["Reason"] = (
            rel_df_reason_drug_count["entity1_entity2"].str.split(":").str[0]
        )
        rel_df_reason_drug_count["Drug"] = (
            rel_df_reason_drug_count["entity1_entity2"].str.split(":").str[1]
        )

        # For each entity that is a 'Drug', join the 'Reason' from rel_df_reason_drug_count using 'Drug' that has the highest count
        for index, row in ent_df.iterrows():
            index = int(index)  # type: ignore
            if row["category"] == "Drug":
                try:
                    join_options = (
                        rel_df_reason_drug_count[
                            rel_df_reason_drug_count["Drug"] == row["text"]
                        ]
                        .sort_values(by="count_in_document", ascending=False)
                        .head(5)["Reason"]
                        .values
                    )
                    # Loop through join_options and find the one with length larger than 4
                    option_found = False
                    for option in join_options:
                        if len(option) > 4:
                            ent_df.loc[index, "Joined_Reason"] = str(option)
                            option_found = True
                            break

                    # If no option is found, join the first option
                    if not option_found:
                        ent_df.loc[index, "Joined_Reason"] = join_options[0]
                except:
                    ent_df.loc[index, "Joined_Reason"] = ""

        # Filter ent_df_reason_drug to only 'Drug' category
        ent_df["entity_type"] = ""
        ent_df_reason_drug = ent_df[ent_df["category"] == "Drug"]
        for i, row in ent_df_reason_drug.iterrows():
            try:
                ent_df_reason_drug.at[i, "entity_type"] = ner_prediction(
                    str(row["Joined_Reason"]), compute="cpu"
                )["entity_group"].iloc[0]
            except:
                ent_df_reason_drug.at[i, "entity_type"] = ""

        self.ent_df = ent_df

    def __parse_top_5_factors(self) -> pd.DataFrame:
        """
        Extracts the top 5 factors for each clinician note

        Returns
        -------
        pandas DataFrame
            Dataframe containing top 5 factors for each clinician note.
        """

        print("Parsing top 5 factors...")
        ent_df_reason_drug = self.ent_df
        # Filter out entities which aren't factors
        ent_df_reason_drug = ent_df_reason_drug[
            ~ent_df_reason_drug["entity_type"].isin(
                ["Therapeutic_procedure", "Lab_value", "Detailed_description"]
            )
        ]
        ent_df_reason_drug = ent_df_reason_drug[
            ent_df_reason_drug["Joined_Reason"].str.len() > 0
        ]

        top_5_factors_by_file = pd.DataFrame()
        for file_idx in self.txt_df["file_idx"].unique():
            common_factors = ",".join(
                ent_df_reason_drug[ent_df_reason_drug["file_idx"] == file_idx]
                .sort_values(by="count_in_document", ascending=False)["Joined_Reason"]
                .unique()
            )
            
            lemmatized_words = []
            factors_cleaned = ""
            counter = 0
            for _, factor in enumerate(common_factors.split(",")):
                if counter == 5:
                    break
                factor_cleaned = factor.strip()
                factor_cleaned = factor.title()
                lemmatized_word = self.lemmatizer.lemmatize(factor_cleaned)
                if lemmatized_word not in lemmatized_words:
                    factors_cleaned += factor_cleaned + ","
                    counter += 1
                    lemmatized_words.append(lemmatized_word)
            top_5_factors_by_file = top_5_factors_by_file.append({"file_idx": file_idx, "Common_Underlying_Factors": factors_cleaned}, ignore_index=True)  # type: ignore

        return top_5_factors_by_file
