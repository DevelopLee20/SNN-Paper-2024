from SpikingNeuralNetwork import Tutorial3_SNN_Runner
import numpy as np
import torch
import logging

Tutorial3_SNN_Runner.set_download_FashionMNIST()
Tutorial3_SNN_Runner.set_pytorch_device()


logging.info(
    "Data Size: ", Tutorial3_SNN_Runner.x_train.shape, Tutorial3_SNN_Runner.x_test.shape
)

seed = 1004
np.random.seed(seed)
torch.manual_seed(seed)

nb_hidden = [[100], [100, 100], [100, 100, 100]]
nb_epochs = [5, 10, 15]

for hidden in nb_hidden:
    for epochs in nb_epochs:
        Tutorial3_SNN_Runner.nb_hidden = hidden
        Tutorial3_SNN_Runner.nb_epochs = epochs
        Tutorial3_SNN_Runner.set_layers_weight_list()
        Tutorial3_SNN_Runner.train(
            Tutorial3_SNN_Runner.x_train, Tutorial3_SNN_Runner.y_train
        )
