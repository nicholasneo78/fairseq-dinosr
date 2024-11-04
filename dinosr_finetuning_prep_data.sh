# # GENERATE TSV, for first 100h of data only
# PATH_TO_WAVES=/datasets/librispeech/train/train-clean-100
# MANIFEST_PATH=/datasets/librispeech/train/train-clean-100
# EXT=flac

# python3 /fairseq_dinosr/examples/wav2vec/wav2vec_manifest.py $PATH_TO_WAVES --dest $MANIFEST_PATH --ext $EXT --valid-percent 0

# # GENERATE FINAL MANIFEST WITH TRANSCRIPTIONS
# SPLIT=train
# PATH_TO_TSV=/datasets/librispeech/train/train-clean-100/train.tsv
# OUTPUT_DIR=/datasets/librispeech/train/train-clean-100

# python3 /fairseq_dinosr/examples/wav2vec/libri_labels.py $PATH_TO_TSV --output-dir $OUTPUT_DIR --output-name $SPLIT

# GENERATE TSV, test and test set
PATH_TO_WAVES=/datasets/librispeech/test-clean
MANIFEST_PATH=/datasets/librispeech/test-clean
EXT=flac

python3 /fairseq_dinosr/examples/wav2vec/wav2vec_manifest.py $PATH_TO_WAVES --dest $MANIFEST_PATH --ext $EXT --valid-percent 0
mv /datasets/librispeech/test-clean/train.tsv /datasets/librispeech/test-clean/test.tsv

# GENERATE FINAL MANIFEST WITH TRANSCRIPTIONS
SPLIT=test
PATH_TO_TSV=/datasets/librispeech/test-clean/test.tsv
OUTPUT_DIR=/datasets/librispeech/test-clean

python3 /fairseq_dinosr/examples/wav2vec/libri_labels.py $PATH_TO_TSV --output-dir $OUTPUT_DIR --output-name $SPLIT