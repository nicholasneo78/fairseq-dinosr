ROOT_MODEL_DIR=/models/dinosr/librispeech_960_bs_10m_steps_2.4m_retrained
# ROOT_MODEL_DIR=/models/dinosr/librispeech_test

mkdir -p $ROOT_MODEL_DIR
python3 fairseq_cli/hydra_train.py -m \
    --config-dir examples/dinosr/config/ \
    --config-name base \
    checkpoint.root_model_dir=$ROOT_MODEL_DIR \
    task.data=/datasets/librispeech/ \
    common.user_dir=examples/dinosr &
