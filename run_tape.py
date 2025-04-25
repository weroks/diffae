from templates import *
from templates_latent import *

if __name__ == "__main__":
    # train the autoenc moodel
    # this requires V100s.
    # gpus = [0, 1, 2, 3]
    gpus = [0]
    conf = tapev06_autoenc()
    train(conf, gpus=gpus)
