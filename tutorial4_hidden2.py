# import os
# import h5py

# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.gridspec import GridSpec
# import seaborn as sns

# import torch
# import torch.nn as nn

# from utils import get_shd_dataset

# # The coarse network structure and the time steps are dicated by the SHD dataset.
# nb_inputs = 700
# nb_hidden = 200
# nb_outputs = 20

# time_step = 1e-3
# nb_steps = 100
# max_time = 1.4

# batch_size = 256

# dtype = torch.float

# # Check whether a GPU is available
# if torch.cuda.is_available():
#     device = torch.device("cuda")
# else:
#     device = torch.device("cpu")

# # Here we load the Dataset
# cache_dir = os.path.expanduser("./data")
# cache_subdir = "hdspikes"
# get_shd_dataset(cache_dir, cache_subdir)

# train_file = h5py.File(os.path.join(cache_dir, cache_subdir, "shd_train.h5"), "r")
# test_file = h5py.File(os.path.join(cache_dir, cache_subdir, "shd_test.h5"), "r")

# x_train = train_file["spikes"]
# y_train = train_file["labels"]
# x_test = test_file["spikes"]
# y_test = test_file["labels"]


# def sparse_data_generator_from_hdf5_spikes(
#     X, y, batch_size, nb_steps, nb_units, max_time, shuffle=True
# ):
#     """This generator takes a spike dataset and generates spiking network input as sparse tensors.

#     Args:
#         X: The data ( sample x event x 2 ) the last dim holds (time,neuron) tuples
#         y: The labels
#     """

#     labels_ = np.array(y, dtype=np.int64)
#     number_of_batches = len(labels_) // batch_size
#     sample_index = np.arange(len(labels_))

#     # compute discrete firing times
#     firing_times = X["times"]
#     units_fired = X["units"]

#     time_bins = np.linspace(0, max_time, num=nb_steps)

#     if shuffle:
#         np.random.shuffle(sample_index)

#     total_batch_count = 0
#     counter = 0
#     while counter < number_of_batches:
#         batch_index = sample_index[batch_size * counter : batch_size * (counter + 1)]

#         coo = [[] for i in range(3)]
#         for bc, idx in enumerate(batch_index):
#             times = np.digitize(firing_times[idx], time_bins)
#             units = units_fired[idx]
#             batch = [bc for _ in range(len(times))]

#             coo[0].extend(batch)
#             coo[1].extend(times)
#             coo[2].extend(units)

#         i = torch.LongTensor(coo).to(device)
#         v = torch.FloatTensor(np.ones(len(coo[0]))).to(device)

#         X_batch = torch.sparse_coo_tensor(
#             i, v, torch.Size([batch_size, nb_steps, nb_units])
#         ).to(device)
#         y_batch = torch.tensor(labels_[batch_index], device=device)

#         yield X_batch.to(device=device), y_batch.to(device=device)

#         counter += 1


# tau_mem = 10e-3
# tau_syn = 5e-3

# alpha = float(np.exp(-time_step / tau_syn))
# beta = float(np.exp(-time_step / tau_mem))

# weight_scale = 0.2

# w1 = torch.empty((nb_inputs, nb_hidden), device=device, dtype=dtype, requires_grad=True)
# torch.nn.init.normal_(w1, mean=0.0, std=weight_scale / np.sqrt(nb_inputs))

# # w2 = torch.empty((nb_hidden, nb_outputs), device=device, dtype=dtype, requires_grad=True)
# # torch.nn.init.normal_(w2, mean=0.0, std=weight_scale/np.sqrt(nb_hidden))

# v1 = torch.empty((nb_hidden, nb_hidden), device=device, dtype=dtype, requires_grad=True)
# torch.nn.init.normal_(v1, mean=0.0, std=weight_scale / np.sqrt(nb_hidden))

# # 새로운 두 번째 은닉층의 가중치 정의
# nb_hidden2 = 100  # 두 번째 은닉층의 뉴런 수

# # 첫 번째 은닉층과 두 번째 은닉층 간의 가중치
# w2 = torch.empty(
#     (nb_hidden, nb_hidden2), device=device, dtype=dtype, requires_grad=True
# )
# torch.nn.init.normal_(w2, mean=0.0, std=weight_scale / np.sqrt(nb_hidden))

# # 두 번째 은닉층과 출력층 간의 가중치
# w3 = torch.empty(
#     (nb_hidden2, nb_outputs), device=device, dtype=dtype, requires_grad=True
# )
# torch.nn.init.normal_(w3, mean=0.0, std=weight_scale / np.sqrt(nb_hidden2))

# # 두 번째 은닉층 내부 연결 가중치
# v2 = torch.empty(
#     (nb_hidden2, nb_hidden2), device=device, dtype=dtype, requires_grad=True
# )
# torch.nn.init.normal_(v2, mean=0.0, std=weight_scale / np.sqrt(nb_hidden2))

# print("init done")


# def plot_voltage_traces(mem, spk=None, dim=(3, 5), spike_height=5):
#     gs = GridSpec(*dim)
#     if spk is not None:
#         dat = 1.0 * mem
#         dat[spk > 0.0] = spike_height
#         dat = dat.detach().cpu().numpy()
#     else:
#         dat = mem.detach().cpu().numpy()
#     for i in range(np.prod(dim)):
#         if i == 0:
#             a0 = ax = plt.subplot(gs[i])
#         else:
#             ax = plt.subplot(gs[i], sharey=a0)
#         ax.plot(dat[i])
#         ax.axis("off")


# from IPython.display import clear_output


# def live_plot(loss):
#     if len(loss) == 1:
#         return
#     clear_output(wait=True)
#     ax = plt.figure(figsize=(3, 2), dpi=150).gca()
#     ax.plot(range(1, len(loss) + 1), loss)
#     ax.set_xlabel("Epoch")
#     ax.set_ylabel("Loss")
#     ax.xaxis.get_major_locator().set_params(integer=True)
#     sns.despine()
#     plt.show()


# class SurrGradSpike(torch.autograd.Function):
#     """
#     Here we implement our spiking nonlinearity which also implements
#     the surrogate gradient. By subclassing torch.autograd.Function,
#     we will be able to use all of PyTorch's autograd functionality.
#     Here we use the normalized negative part of a fast sigmoid
#     as this was done in Zenke & Ganguli (2018).
#     """

#     scale = 100.0  # controls steepness of surrogate gradient

#     @staticmethod
#     def forward(ctx, input):
#         """
#         In the forward pass we compute a step function of the input Tensor
#         and return it. ctx is a context object that we use to stash information which
#         we need to later backpropagate our error signals. To achieve this we use the
#         ctx.save_for_backward method.
#         """
#         ctx.save_for_backward(input)
#         out = torch.zeros_like(input)
#         out[input > 0] = 1.0
#         return out

#     @staticmethod
#     def backward(ctx, grad_output):
#         """
#         In the backward pass we receive a Tensor we need to compute the
#         surrogate gradient of the loss with respect to the input.
#         Here we use the normalized negative part of a fast sigmoid
#         as this was done in Zenke & Ganguli (2018).
#         """
#         (input,) = ctx.saved_tensors
#         grad_input = grad_output.clone()
#         grad = grad_input / (SurrGradSpike.scale * torch.abs(input) + 1.0) ** 2
#         return grad


# # here we overwrite our naive spike function by the "SurrGradSpike" nonlinearity which implements a surrogate gradient
# spike_fn = SurrGradSpike.apply


# def run_snn_2(inputs):
#     # 첫 번째 은닉층 변수
#     syn1 = torch.zeros((batch_size, nb_hidden), device=device, dtype=dtype)
#     mem1 = torch.zeros((batch_size, nb_hidden), device=device, dtype=dtype)

#     # 두 번째 은닉층 변수
#     syn2 = torch.zeros((batch_size, nb_hidden2), device=device, dtype=dtype)
#     mem2 = torch.zeros((batch_size, nb_hidden2), device=device, dtype=dtype)

#     mem_rec1 = []
#     spk_rec1 = []

#     mem_rec2 = []
#     spk_rec2 = []

#     # 첫 번째 은닉층의 활동
#     out1 = torch.zeros((batch_size, nb_hidden), device=device, dtype=dtype)
#     h1_from_input = torch.einsum("abc,cd->abd", (inputs, w1))

#     for t in range(nb_steps):
#         h1 = h1_from_input[:, t] + torch.einsum("ab,bc->ac", (out1, v1))
#         mthr1 = mem1 - 1.0
#         out1 = spike_fn(mthr1)
#         rst1 = out1.detach()  # 리셋에서는 역전파 제외

#         new_syn1 = alpha * syn1 + h1
#         new_mem1 = (beta * mem1 + syn1) * (1.0 - rst1)

#         mem_rec1.append(mem1)
#         spk_rec1.append(out1)

#         mem1 = new_mem1
#         syn1 = new_syn1

#     mem_rec1 = torch.stack(mem_rec1, dim=1)
#     spk_rec1 = torch.stack(spk_rec1, dim=1)

#     # 두 번째 은닉층의 활동
#     out2 = torch.zeros((batch_size, nb_hidden2), device=device, dtype=dtype)
#     h2_from_hidden1 = torch.einsum("abc,cd->abd", (spk_rec1, w2))

#     for t in range(nb_steps):
#         h2 = h2_from_hidden1[:, t] + torch.einsum("ab,bc->ac", (out2, v2))
#         mthr2 = mem2 - 1.0
#         out2 = spike_fn(mthr2)
#         rst2 = out2.detach()

#         new_syn2 = alpha * syn2 + h2
#         new_mem2 = (beta * mem2 + syn2) * (1.0 - rst2)

#         mem_rec2.append(mem2)
#         spk_rec2.append(out2)

#         mem2 = new_mem2
#         syn2 = new_syn2

#     mem_rec2 = torch.stack(mem_rec2, dim=1)
#     spk_rec2 = torch.stack(spk_rec2, dim=1)

#     # 출력층
#     h3 = torch.einsum("abc,cd->abd", (spk_rec2, w3))
#     flt = torch.zeros((batch_size, nb_outputs), device=device, dtype=dtype)
#     out = torch.zeros((batch_size, nb_outputs), device=device, dtype=dtype)
#     out_rec = [out]

#     for t in range(nb_steps):
#         new_flt = alpha * flt + h3[:, t]
#         new_out = beta * out + flt

#         flt = new_flt
#         out = new_out

#         out_rec.append(out)

#     out_rec = torch.stack(out_rec, dim=1)
#     other_recs = [mem_rec1, spk_rec1, mem_rec2, spk_rec2]

#     return out_rec, other_recs


# def train_2(x_data, y_data, lr=1e-3, nb_epochs=10):
#     # 두 번째 은닉층 관련 가중치 (w3, v2)를 추가
#     params = [w1, w2, w3, v1, v2]
#     optimizer = torch.optim.Adamax(params, lr=lr, betas=(0.9, 0.999))

#     log_softmax_fn = nn.LogSoftmax(dim=1)
#     loss_fn = nn.NLLLoss()

#     loss_hist = []
#     for e in range(nb_epochs):
#         local_loss = []
#         for x_local, y_local in sparse_data_generator_from_hdf5_spikes(
#             x_data, y_data, batch_size, nb_steps, nb_inputs, max_time
#         ):
#             output, recs = run_snn_2(x_local.to_dense())

#             mem1_rec, spk1_rec, mem2_rec, spk2_rec = recs

#             m, _ = torch.max(output, 1)
#             log_p_y = log_softmax_fn(m)

#             # 첫 번째 은닉층의 스파이크 기록을 사용한 정규화 손실 계산
#             reg_loss = 2e-6 * torch.sum(spk1_rec)
#             reg_loss += 2e-6 * torch.mean(
#                 torch.sum(torch.sum(spk1_rec, dim=0), dim=0) ** 2
#             )

#             # 두 번째 은닉층 스파이크 기록도 정규화 손실에 포함할 수 있음
#             reg_loss += 2e-6 * torch.sum(spk2_rec)
#             reg_loss += 2e-6 * torch.mean(
#                 torch.sum(torch.sum(spk2_rec, dim=0), dim=0) ** 2
#             )

#             loss_val = loss_fn(log_p_y, y_local) + reg_loss

#             optimizer.zero_grad()
#             loss_val.backward()
#             optimizer.step()
#             local_loss.append(loss_val.item())
#         mean_loss = np.mean(local_loss)
#         loss_hist.append(mean_loss)
#         live_plot(loss_hist)
#         print(f"Epoch {e+1}: loss={mean_loss:.5f}")

#     return loss_hist


# def compute_classification_accuracy(x_data, y_data):
#     """Computes classification accuracy on supplied data in batches."""
#     accs = []
#     for x_local, y_local in sparse_data_generator_from_hdf5_spikes(
#         x_data, y_data, batch_size, nb_steps, nb_inputs, max_time, shuffle=False
#     ):
#         output, _ = run_snn(x_local.to_dense())
#         m, _ = torch.max(output, 1)  # max over time
#         _, am = torch.max(m, 1)  # argmax over output units
#         tmp = np.mean((y_local == am).detach().cpu().numpy())  # compare to labels
#         accs.append(tmp)
#     return np.mean(accs)


# nb_epochs = 200
# # loss_hist = train(x_train, y_train, lr=2e-4, nb_epochs=nb_epochs)/
