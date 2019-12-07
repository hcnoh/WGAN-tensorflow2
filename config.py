LATENT_DEPTH = 100

BATCH_SIZE = 64
NUM_EPOCHS = 100

N_CRITIC = 5
CLIP_CONST = 0.01

MODEL_SAVE_DIR = "../../learned-models/WGAN-tf2"
MODEL_NAME = "WGAN"

DATASET = "mnist"
HYPARAMS = {
    "mnist": {
        "project_shape": [7, 7, 256],
        "gen_filters_list": [128, 64, 1],
        "gen_strides_list": [1, 2, 2],
        "disc_filters_list": [64, 128],
        "disc_strides_list": [2, 2]
    }
}