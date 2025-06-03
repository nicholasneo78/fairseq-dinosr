"""
To retrieve all the unique tokens from the manifest files to form the dict.ltr.txt from the .wrd file for fairseq processing
"""
import os

class CharacterTokenizer:
    def __init__(self, train_path: str, dev_path: str):
        self.train_path = train_path
        self.dev_path = dev_path
        self.unique_chars = set()

    def extract_unique_characters(self):
        for file_path in [self.train_path, self.dev_path]:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    self.unique_chars.update(line.strip())
        # Use | for space
        if ' ' in self.unique_chars:
            self.unique_chars.remove(' ')
        self.unique_chars = ['|'] + sorted(self.unique_chars)

    def save_token_file(self, output_path: str):
        self.extract_unique_characters()
        with open(output_path, 'w', encoding='utf-8') as out_file:
            for token_id, char in enumerate(self.unique_chars):
                out_file.write(f"{char} {token_id}\n")


if __name__ == "__main__":

    # Example usage:
    # tokenizer = CharacterTokenizer('train.wrd', 'dev.wrd')
    # tokenizer.save_token_file('char_tokens.txt')

    # ROOT = '/datasets/fleurs_p1/th'
    ROOT = '/nas/asr_50h'
    tokenizer = CharacterTokenizer(
        train_path=os.path.join(ROOT, 'train.wrd'),
        dev_path=os.path.join(ROOT, 'dev.wrd')
    )


    tokenizer.save_token_file(output_path=os.path.join(ROOT, "dict.ltr.txt"))