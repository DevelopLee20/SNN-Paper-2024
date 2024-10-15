from SpikingNeuralNetwork import Tutorial3_SNN_Runner

Tutorial3_SNN_Runner.set_download_FashionMNIST()

print(
    "Data Size: ", Tutorial3_SNN_Runner.x_train.shape, Tutorial3_SNN_Runner.x_test.shape
)

print("run one")
Tutorial3_SNN_Runner.set_pytorch_device()
Tutorial3_SNN_Runner.set_layers_weight_list()
Tutorial3_SNN_Runner.nb_hidden = [100, 100]
Tutorial3_SNN_Runner.nb_epochs = 5
result_loss = Tutorial3_SNN_Runner.train(
    Tutorial3_SNN_Runner.x_train, Tutorial3_SNN_Runner.y_train
)
print(result_loss)
