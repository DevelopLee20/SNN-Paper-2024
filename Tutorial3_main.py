from SpikingNeuralNetwork import Tutorial3_SNN_Runner

x_train, y_train, x_test, y_test = Tutorial3_SNN_Runner.get_download_FashionMNIST()

print("Data Size: ", x_train.shape, x_test.shape)

print("run one")
Tutorial3_SNN_Runner.weight_list = Tutorial3_SNN_Runner.get_layers_weight_list()
result_loss = Tutorial3_SNN_Runner.train(x_train, y_train)
print(result_loss)

print("run two")
Tutorial3_SNN_Runner.weight_list = Tutorial3_SNN_Runner.get_layers_weight_list()
Tutorial3_SNN_Runner.nb_hidden = [100, 100]
result_loss = Tutorial3_SNN_Runner.train(x_train, y_train)
print(result_loss)
