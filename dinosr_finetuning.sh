ROOT_MODEL_DIR=/models/dinosr/librispeech_960_bs_10m/finetuned_librispeech_100_bs_6.4m

mkdir -p $ROOT_MODEL_DIR
python3 fairseq_cli/hydra_train.py -m \
    --config-dir examples/wav2vec/config/finetuning \
    --config-name base_100h \
    checkpoint.root_model_dir=$ROOT_MODEL_DIR \
    common.user_dir=examples/dinosr \
    task.data=/datasets/librispeech/finetuning_config  \
    model.w2v_path=/models/dinosr/librispeech_960_bs_10m/ckpt/checkpoint_72_400000.pt \
    task.normalize=True