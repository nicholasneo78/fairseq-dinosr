# ROOT_MODEL_DIR=/models/dinosr/multilingual # /librispeech_default_pretrained/finetuned_librispeech_100_bs_6.4m/ckpt
# ROOT_MODEL=${ROOT_MODEL_DIR}/finetune_fleurs_p1_langs.pt  # wav2vec_small_960h.pt
# ROOT_DATA_DIR=/datasets/librispeech/finetuning_config

# python3 examples/speech_recognition/infer.py $ROOT_DATA_DIR --task audio_finetuning \
#     --path $ROOT_MODEL --gen-subset test --results-path $ROOT_MODEL_DIR \
#     --criterion ctc --labels ltr --max-tokens 4000000 \
#     --post-process letter

# python examples/speech_recognition/infer.py /checkpoint/abaevski/data/speech/libri/10h/wav2vec/raw --task audio_finetuning \
# --nbest 1 --path /path/to/model --gen-subset $subset --results-path /path/to/save/results/for/sclite --w2l-decoder kenlm \
# --lm-model /path/to/kenlm.bin --lm-weight 2 --word-score -1 --sil-weight 0 --criterion ctc --labels ltr --max-tokens 4000000 \
# --post-process letter

# eval_data_root=/datasets/librispeech/finetuning_config
# log_dir=/models/dinosr/librispeech/finetuned_librispeech_100_bs_6.4m/eval
# ft_ckpt=/models/dinosr/librispeech/finetuned_librispeech_100_bs_6.4m/ckpt/checkpoint_best.pt

# eval_data_root=/datasets/fleurs_p1/th/fleurs_test
# log_dir=/models/dinosr/multilingual/eval_th
# ft_ckpt=/models/dinosr/multilingual/finetuned_malt_50h_5lang/ckpt/checkpoint_best.pt

eval_data_root=/datasets/fleurs_p1/tl/fleurs_test
log_dir=/models/dinosr/multilingual/eval_tl
ft_ckpt=/models/dinosr/multilingual/finetune_fleurs_p1_langs.pt

export PYTHONPATH=$(pwd):$PYTHONPATH

python3 /fairseq_dinosr/examples/speech_recognition/new/infer.py -m \
    --config-dir /fairseq_dinosr/examples/wav2vec/config/finetuning \
    --config-name infer_viterbi \
    common.user_dir=/fairseq_dinosr/examples/dinosr \
    task.data=$eval_data_root \
    task.normalize=true \
    common_eval.results_path=$log_dir \
    common_eval.path=$ft_ckpt \
    common_eval.post_process=none \
    dataset.gen_subset=test \
    dataset.batch_size=16 \
    decoding.results_path=$log_dir \
    decoding.prediction_log_path=$log_dir/predictions.json