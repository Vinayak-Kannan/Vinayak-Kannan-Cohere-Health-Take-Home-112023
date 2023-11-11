import glob
import pandas as pd
from time import time
from typing import Tuple


def load_txt(data_path: str) -> pd.DataFrame:
    """
    Load the text from the .txt files as a df.
    
    Parameters
    ----------
    data_path: str
        Path to location of .txt files.
    
    
    Returns
    -------
    pandas DataFrame
        Df indexed by file name containing the text extracted 
        from .txt files.
    """
    # load txt files
    t0 = time()
    
    file_paths = glob.glob(data_path + "*.txt")
    files = []
    for i in file_paths:
        with open(i, "r") as file:
            l = file.read()
            files.append(l)

    print(f"Time taken to read .txt files: {time() - t0}")
    
    # convert to df
    file_names = [p.split("/")[-1].split(".")[0] for p in file_paths]
    text_df = pd.DataFrame({
        "file_idx": file_names,
        "text": files
    })
    
    return text_df


def load_ann(data_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    The .ann files contain metadata on entities present in the clinical
    documentation as well as some relationships between some of the
    entities. Load in this metadata from .ann files as two dfs.
    
    Parameters
    ----------
    data_path: str
        Path to location of .ann files.
        
    Returns
    -------
    Tuple[pandas DataFrame, pandas DataFrame]
        Two DFs indexed by file name. One contains all the labeled entities
        from the .txt files. The other contains the relationships between
        some of the entities.
    """
    # load ann files
    t0 = time()

    ann_paths = glob.glob(data_path + "*.ann") 
    entity_dict = {
        "file_idx": [],
        "entity_id": [],
        "category": [],
        "start_idx": [],
        "end_idx": [],
        "text": []
    }

    rel_dict = {
        "file_idx": [],
        "relationship_id": [],
        "category": [],
        "entity1": [],
        "entity2": []
    }

    for i in ann_paths:
        file_idx = i.split("/")[-1].split(".")[0]
        with open(i, "r") as f:
            lines = f.readlines()
            # reasons.append(lines)

        for line in lines:
            out = line.split("\t")
            ann_type = out[0][0]

            if ann_type == "T":
                # entities
                entity_dict["file_idx"].append(file_idx)
                entity_dict["entity_id"].append(out[0])
                entity_dict["category"].append(out[1].split(" ")[0])
                entity_dict["start_idx"].append(out[1].split(" ")[1])
                entity_dict["end_idx"].append(out[1].split(" ")[2])
                entity_dict["text"].append(out[2])

            elif ann_type == "R":
                # relationships
                rel_dict["file_idx"].append(file_idx)
                rel_dict["relationship_id"].append(out[0])
                rel_dict["category"].append(out[1].split(" ")[0])
                rel_dict["entity1"].append(out[1].split(" ")[1])
                rel_dict["entity2"].append(out[1].split(" ")[2])

            else:
                # skip annotator notes
                continue

    print("Time taken to read .ann files and extract all " + 
          f"metadata: {time() - t0}")
    
    ann_df = pd.DataFrame(entity_dict)
    rel_df = pd.DataFrame(rel_dict)
    
    return ann_df, rel_df
