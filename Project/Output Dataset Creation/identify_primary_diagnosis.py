import spacy
import pandas as pd
import nltk


class PrimaryDiagnosisIdentifier:
    txt_df = pd.DataFrame()
    output_df = pd.DataFrame(columns=["file_idx", "primary_diagnosis", "count"])

    def __init__(self, txt_df: pd.DataFrame) -> None:
        self.txt_df = txt_df

    def process_data(self, txt_df: pd.DataFrame) -> pd.DataFrame:
        print("Processing Data...")
        self.output_df = self.__create_primary_diagnosis_freq_df(txt_df)
        return self.output_df

    def __create_primary_diagnosis_freq_df(self, txt_df: pd.DataFrame) -> pd.DataFrame:
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
                local_output_df = local_output_df.append({"file_idx": file_idx, "primary_diagnosis": key, "count": value}, ignore_index=True)  # type: ignore

        return local_output_df
