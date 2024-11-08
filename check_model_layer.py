import torch

MODEL="/models/dinosr/wav2vec_small_960h.pt"

model = torch.load(MODEL)

for key in model:
    if "_name" in key:
        print(key, model[key])
    else:
        print("no name")