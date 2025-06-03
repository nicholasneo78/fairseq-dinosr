import os
import json
from tqdm import tqdm
from typing import List, Dict
import pandas as pd

def load_manifest_nemo(input_manifest_path: str) -> List[Dict[str, str]]:

    '''
    loads the manifest file in Nvidia NeMo format to process the entries and store them into a list of dictionaries

    the manifest file would contain entries in this format:

    {"audio_filepath": "subdir1/xxx1.wav", "duration": 3.0, "text": "shan jie is an orange cat"}
    {"audio_filepath": "subdir1/xxx2.wav", "duration": 4.0, "text": "shan jie's orange cat is chonky"}
    ---

    input_manifest_path: the manifest path that contains the information of the audio clips of interest
    ---
    returns: a list of dictionaries of the information in the input manifest file
    '''

    dict_list = []

    with open(input_manifest_path, 'r+') as f:
        for line in f:
            dict_list.append(json.loads(line))

    return dict_list

class ConvertNemoToTSV:

    def __init__(
        self,
        input_manifest_dir: str,
        output_manifest_dir: str,
        output_root_dir: str,
        sampling_rate: int=16000,
    ) -> None:
        
        """
        input_manifest_dir: input manifest dir in the nemo format
        output_manifest_dir: output manifest dir in the tsv format that fairseq will take in
        output_root_dir: the output root dir that fairseq takes in the tsv file
        """

        self.input_manifest_dir = input_manifest_dir
        self.output_manifest_dir = output_manifest_dir
        self.output_root_dir = output_root_dir
        self.sampling_rate = sampling_rate

    def convert(self) -> None:

        final_output_list = []

        items = load_manifest_nemo(input_manifest_path=self.input_manifest_dir)

        for item in tqdm(items):
            temp = {
                "audio_filepath": item['audio_filepath'],
                "num_samples": int(item['duration'] * self.sampling_rate)
            }

            final_output_list.append(temp)

        # convert the list of dict into a dataframe
        df = pd.DataFrame(final_output_list)

        # insert the root at the top of the df
        new_row = [""] * len(df.columns)
        new_row[0] = self.output_root_dir
        new_df = pd.DataFrame([new_row], columns=df.columns)
        df = pd.concat([new_df, df], ignore_index=True)

        df.columns = df.iloc[0]    # Set the first row as header
        df = df[1:]                # Remove the first row from the data
        df = df.reset_index(drop=True)

        # saving as tsv file
        df.to_csv(self.output_manifest_dir, sep="\t", index=False)

    def __call__(self) -> None:
        return self.convert()
    

if __name__ == "__main__":

    # ROOT = '/datasets/fleurs_p1/th/fleurs_test'
    ROOT = "/nas/asr_50h"

    INPUT_MANIFEST_DIR = os.path.join(ROOT, 'test_manifest.json')
    OUTPUT_MANIFEST_DIR = os.path.join(ROOT, 'test.tsv')
    SR = 16000

    c = ConvertNemoToTSV(
        input_manifest_dir=INPUT_MANIFEST_DIR,
        output_manifest_dir=OUTPUT_MANIFEST_DIR,
        output_root_dir=ROOT,
        sampling_rate=SR
    )()

    INPUT_MANIFEST_DIR = os.path.join(ROOT, 'train_manifest.json')
    OUTPUT_MANIFEST_DIR = os.path.join(ROOT, 'train.tsv')
    SR = 16000

    c = ConvertNemoToTSV(
        input_manifest_dir=INPUT_MANIFEST_DIR,
        output_manifest_dir=OUTPUT_MANIFEST_DIR,
        output_root_dir=ROOT,
        sampling_rate=SR
    )()

    INPUT_MANIFEST_DIR = os.path.join(ROOT, 'dev_manifest.json')
    OUTPUT_MANIFEST_DIR = os.path.join(ROOT, 'dev.tsv')
    SR = 16000

    c = ConvertNemoToTSV(
        input_manifest_dir=INPUT_MANIFEST_DIR,
        output_manifest_dir=OUTPUT_MANIFEST_DIR,
        output_root_dir=ROOT,
        sampling_rate=SR
    )()