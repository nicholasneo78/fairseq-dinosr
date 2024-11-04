# this code is to be ran outside of the docker container
# MAKE SURE THE dinosr_pretraining.sh FILE HAS BEEN CONFIGURED PROPERLY, ESPECIALLY THE MODEL PATH
# MAKE SURE THE base.yaml FILE HAS BEEN CONFIGURED PROPERLY TOO

# have to spin up the tensorboard first
echo LAUNCHING TENSORBOARD
docker run -it -d -p 6006:6006 -v /home/dh/models/dinosr:/logs fairseq_dinosr:v0.0.1 tensorboard --logdir /logs --host 0.0.0.0 --port 6006

# actual execution of the experiment
echo START EXPERIMENT
docker run --name fairseq_dinosr -it -d --gpus 1 -v /home/dh/code/fairseq-dinosr/:/fairseq_dinosr -v /home/dh/datasets/:/datasets -v /home/dh/models/:/models --ulimit memlock=-1 --ulimit stack=67108864 --shm-size 32gb fairseq_dinosr:v0.0.1 bash -c "./dinosr_pretraining.sh && tail -F anything"

# retrieve the container that was spin for dinosr
docker ps | grep fairseq_dinosr