# ROOT_MODEL_DIR=/models/dinosr/librispeech_960_bs_10m
# PRETRAINED_MODEL_DIR=${ROOT_MODEL_DIR}/ckpt/checkpoint_72_400000.pt
# FINETUNED_MODEL_DIR=${ROOT_MODEL_DIR}/finetuned_librispeech_100_bs_6.4m

ROOT_MODEL_DIR=/models/dinosr/librispeech_default_pretrained
PRETRAINED_MODEL_DIR=${ROOT_MODEL_DIR}/dinosr.ckpt
FINETUNED_MODEL_DIR=${ROOT_MODEL_DIR}/finetuned_librispeech_100_bs_6.4m

mkdir -p $FINETUNED_MODEL_DIR
python3 fairseq_cli/hydra_train.py -m \
    --config-dir examples/wav2vec/config/finetuning \
    --config-name base_100h \
    checkpoint.root_model_dir=$FINETUNED_MODEL_DIR \
    common.user_dir=examples/dinosr \
    task.data=/datasets/librispeech/finetuning_config  \
    model.w2v_path=$PRETRAINED_MODEL_DIR \
    task.normalize=True