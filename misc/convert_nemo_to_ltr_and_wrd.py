import os
import json
from tqdm import tqdm
from typing import List, Dict

from modules import TextPostProcessingManager

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

class ConvertNemoToLtrAndWrd:

    def __init__(
        self,
        input_manifest_dir: str,
        output_ltr_dir: str,
        output_wrd_dir: str,
    ) -> None:
        
        """
        input_manifest_dir: input manifest dir in the nemo format
        output_ltr_dir: output ltr dir for fairseq to process
        output_wrd_dir: output wrd dir for fairseq to process
        """

        self.input_manifest_dir = input_manifest_dir
        self.output_ltr_dir = output_ltr_dir
        self.output_wrd_dir = output_wrd_dir


    def convert_sentence_to_char(self, text: str) -> str:

        formatted = " | ".join(" ".join(word) for word in text.split())
        
        return formatted
        

    def convert(self) -> None:

        ltr_list, wrd_list = [], []

        items = load_manifest_nemo(input_manifest_path=self.input_manifest_dir)

        for item in tqdm(items):
            cleaned_text = TextPostProcessingManager(label=item['language']).process_data(text=item['text'])
            char_text = self.convert_sentence_to_char(text=cleaned_text)
            wrd_list.append(cleaned_text)
            ltr_list.append(char_text)

        # export the text files
        with open(self.output_wrd_dir, 'w') as fw:
            for line in wrd_list:
                fw.write(f"{line}\n")

        with open(self.output_ltr_dir, 'w') as fw:
            for line in ltr_list:
                fw.write(f"{line}\n")

    def __call__(self) -> None:

        return self.convert()
    

if __name__ == "__main__":

    # ROOT = '/datasets/fleurs_p1/th/fleurs_test'
    ROOT = '/nas/asr_50h'

    INPUT_MANIFEST_DIR = os.path.join(ROOT, 'test_manifest.json')
    OUTPUT_LTR_DIR= os.path.join(ROOT, "test.ltr")
    OUTPUT_WRD_DIR= os.path.join(ROOT, "test.wrd")

    c = ConvertNemoToLtrAndWrd(
        input_manifest_dir=INPUT_MANIFEST_DIR,
        output_ltr_dir=OUTPUT_LTR_DIR,
        output_wrd_dir=OUTPUT_WRD_DIR,
    )()

    INPUT_MANIFEST_DIR = os.path.join(ROOT, 'train_manifest.json')
    OUTPUT_LTR_DIR= os.path.join(ROOT, "train.ltr")
    OUTPUT_WRD_DIR= os.path.join(ROOT, "train.wrd")

    c = ConvertNemoToLtrAndWrd(
        input_manifest_dir=INPUT_MANIFEST_DIR,
        output_ltr_dir=OUTPUT_LTR_DIR,
        output_wrd_dir=OUTPUT_WRD_DIR,
    )()

    INPUT_MANIFEST_DIR = os.path.join(ROOT, 'dev_manifest.json')
    OUTPUT_LTR_DIR= os.path.join(ROOT, "dev.ltr")
    OUTPUT_WRD_DIR= os.path.join(ROOT, "dev.wrd")

    c = ConvertNemoToLtrAndWrd(
        input_manifest_dir=INPUT_MANIFEST_DIR,
        output_ltr_dir=OUTPUT_LTR_DIR,
        output_wrd_dir=OUTPUT_WRD_DIR,
    )()