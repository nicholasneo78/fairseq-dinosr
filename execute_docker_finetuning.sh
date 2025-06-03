# this code is to be ran outside of the docker container
# MAKE SURE THE dinosr_pretraining.sh FILE HAS BEEN CONFIGURED PROPERLY, ESPECIALLY THE MODEL PATH
# MAKE SURE THE base.yaml FILE HAS BEEN CONFIGURED PROPERLY TOO

# have to spin up the tensorboard first
echo LAUNCHING TENSORBOARD
docker run --name tensorboard -it -d -p 6006:6006 -v $HOME/models/dinosr:/logs fairseq_dinosr:v0.0.2 tensorboard --logdir /logs --host 0.0.0.0 --port 6006

# actual execution of the experiment
echo START EXPERIMENT
docker run --name fairseq_dinosr_finetuning -it -d --gpus 2 -v $HOME/code/fairseq-dinosr/:/fairseq_dinosr -v $HOME/datasets/:/datasets -v $HOME/models/:/models --ulimit memlock=-1 --ulimit stack=67108864 --shm-size 32gb fairseq_dinosr:v0.0.2 bash -c "CUDA_VISIBLE_DEVICES=0 ./dinosr_finetuning.sh && tail -F anything"

# retrieve the container that was spin for dinosr
docker ps | grep fairseq_dinosr