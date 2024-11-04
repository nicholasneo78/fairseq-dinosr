PATH_TO_WAVES=/datasets/librispeech/train
MANIFEST_PATH=/datasets/librispeech
EXT=flac

python3 /fairseq_dinosr/examples/wav2vec/wav2vec_manifest.py $PATH_TO_WAVES --dest $MANIFEST_PATH --ext $EXT --valid-percent 0