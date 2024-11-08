ROOT_MODEL_DIR=/models/dinosr # /librispeech_default_pretrained/finetuned_librispeech_100_bs_6.4m/ckpt
ROOT_MODEL=${ROOT_MODEL_DIR}/wav2vec_small_960h.pt
ROOT_DATA_DIR=/datasets/librispeech/finetuning_config

python3 examples/speech_recognition/infer.py $ROOT_DATA_DIR --task audio_finetuning \
    --path $ROOT_MODEL --gen-subset test --results-path $ROOT_MODEL_DIR \
    --criterion ctc --labels ltr --max-tokens 4000000 \
    --post-process letter

# python examples/speech_recognition/infer.py /checkpoint/abaevski/data/speech/libri/10h/wav2vec/raw --task audio_finetuning \
# --nbest 1 --path /path/to/model --gen-subset $subset --results-path /path/to/save/results/for/sclite --w2l-decoder kenlm \
# --lm-model /path/to/kenlm.bin --lm-weight 2 --word-score -1 --sil-weight 0 --criterion ctc --labels ltr --max-tokens 4000000 \
# --post-process letter