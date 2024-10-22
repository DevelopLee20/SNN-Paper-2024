import os
import h5py

import numpy as np
import logging
import time
import csv


import torch
import torch.nn as nn

from utils import get_shd_dataset
from SurrogateGradient import SurrogateGradientSpike

# 기본 설정
logging.basicConfig(
    level=logging.DEBUG,  # 로그 레벨 설정
    format="%(asctime)s - %(levelname)s - %(message)s",  # 로그 메시지 형식
    handlers=[
        logging.FileHandler("Tutorial4_runner.log"),  # 로그를 기록할 파일 설정
        logging.StreamHandler(),  # 콘솔에 로그 출력
    ],
)


class VoiceData_Tutorial_4:
    # 하이퍼파라미터
    nb_hidden = [200]  # 은닉층
    nb_steps = 100  # 시간스텝
    surrogate_gradient_scale = 100.0  # 서로게이트 기울기 가파른 정도

    # Normal
    time_step = 1e-3  # 시간 단위
    nb_inputs = 700  # 입력 노드
    nb_outputs = 20  # 출력 노드
    max_time = 1.4  # 음성 최대 시간
    batch_size = 256  # 배치크기
    dtype = torch.float  # 데이터 형식
    tau_mem = 10e-3  # 막전위 변화율
    tau_syn = 5e-3  # 시냅스 변화율
    alpha = float(np.exp(-time_step / tau_syn))  # 파라미터
    beta = float(np.exp(-time_step / tau_mem))  # 파라미터
    weight_scale = 0.2  # 가중치 스케일

    # Need setter once
    device = None
    x_train = None
    y_train = None
    x_test = None
    y_test = None

    # 하나의 학습마다 초기화
    w1 = None
    w2 = None
    w3 = None
    v1 = None
    v2 = None
    spike_fn = None

    @classmethod
    def run_snn_1(cls, inputs):
        syn = torch.zeros(
            (cls.batch_size, cls.nb_hidden[0]), device=cls.device, dtype=cls.dtype
        )
        mem = torch.zeros(
            (cls.batch_size, cls.nb_hidden[0]), device=cls.device, dtype=cls.dtype
        )

        mem_rec = []
        spk_rec = []

        # Compute hidden layer activity
        out = torch.zeros(
            (cls.batch_size, cls.nb_hidden[0]), device=cls.device, dtype=cls.dtype
        )
        h1_from_input = torch.einsum("abc,cd->abd", (inputs, cls.w1))
        for t in range(cls.nb_steps):
            h1 = h1_from_input[:, t] + torch.einsum("ab,bc->ac", (out, cls.v1))
            mthr = mem - 1.0
            out = cls.spike_fn(mthr)
            rst = out.detach()  # We do not want to backprop through the reset

            new_syn = cls.alpha * syn + h1
            new_mem = (cls.beta * mem + syn) * (1.0 - rst)

            mem_rec.append(mem)
            spk_rec.append(out)

            mem = new_mem
            syn = new_syn

        mem_rec = torch.stack(mem_rec, dim=1)
        spk_rec = torch.stack(spk_rec, dim=1)

        # Readout layer
        h2 = torch.einsum("abc,cd->abd", (spk_rec, cls.w2))
        flt = torch.zeros(
            (cls.batch_size, cls.nb_outputs), device=cls.device, dtype=cls.dtype
        )
        out = torch.zeros(
            (cls.batch_size, cls.nb_outputs), device=cls.device, dtype=cls.dtype
        )
        out_rec = [out]
        for t in range(cls.nb_steps):
            new_flt = cls.alpha * flt + h2[:, t]
            new_out = cls.beta * out + flt

            flt = new_flt
            out = new_out

            out_rec.append(out)

        out_rec = torch.stack(out_rec, dim=1)
        other_recs = [mem_rec, spk_rec]
        return out_rec, other_recs

    @classmethod
    def train_1(cls, x_data, y_data, lr=1e-3, nb_epochs=10):
        params = [cls.w1, cls.w2, cls.v1]
        optimizer = torch.optim.Adamax(params, lr=lr, betas=(0.9, 0.999))

        log_softmax_fn = nn.LogSoftmax(dim=1)
        loss_fn = nn.NLLLoss()

        start = time.time()
        loss_hist = []
        for e in range(nb_epochs):
            local_loss = []
            for x_local, y_local in cls.sparse_data_generator_from_hdf5_spikes(
                x_data,
                y_data,
                cls.batch_size,
                cls.nb_steps,
                cls.nb_inputs,
                cls.max_time,
            ):
                output, recs = cls.run_snn_1(x_local.to_dense())
                _, spks = recs
                m, _ = torch.max(output, 1)
                log_p_y = log_softmax_fn(m)

                # Here we set up our regularizer loss
                # The strength paramters here are merely a guess and there should be ample room for improvement by
                # tuning these paramters.
                reg_loss = 2e-6 * torch.sum(spks)  # L1 loss on total number of spikes
                reg_loss += 2e-6 * torch.mean(
                    torch.sum(torch.sum(spks, dim=0), dim=0) ** 2
                )  # L2 loss on spikes per neuron

                # Here we combine supervised loss and the regularizer
                loss_val = loss_fn(log_p_y, y_local) + reg_loss

                optimizer.zero_grad()
                loss_val.backward()
                optimizer.step()
                local_loss.append(loss_val.item())
            mean_loss = np.mean(local_loss)
            loss_hist.append(mean_loss)
            logging.info(f"Epoch: {e+1} loss={mean_loss}")

        end = time.time()

        runningtime = end - start

        return loss_hist, runningtime

    @classmethod
    def set_spike_fn(cls):
        SurrogateGradientSpike.scale = cls.surrogate_gradient_scale
        cls.spike_fn = SurrogateGradientSpike.apply

    @classmethod
    def set_weights_1(cls):
        cls.w1 = torch.empty(
            (cls.nb_inputs, cls.nb_hidden[0]),
            device=cls.device,
            dtype=cls.dtype,
            requires_grad=True,
        )
        torch.nn.init.normal_(
            cls.w1, mean=0.0, std=cls.weight_scale / np.sqrt(cls.nb_inputs)
        )

        cls.w2 = torch.empty(
            (cls.nb_hidden[0], cls.nb_outputs),
            device=cls.device,
            dtype=cls.dtype,
            requires_grad=True,
        )
        torch.nn.init.normal_(
            cls.w2, mean=0.0, std=cls.weight_scale / np.sqrt(cls.nb_hidden[0])
        )

        cls.v1 = torch.empty(
            (cls.nb_hidden[0], cls.nb_hidden[0]),
            device=cls.device,
            dtype=cls.dtype,
            requires_grad=True,
        )
        torch.nn.init.normal_(
            cls.v1, mean=0.0, std=cls.weight_scale / np.sqrt(cls.nb_hidden[0])
        )

    @classmethod
    def set_weights_2(cls):
        cls.w1 = torch.empty(
            (cls.nb_inputs, cls.nb_hidden[0]),
            device=cls.device,
            dtype=cls.dtype,
            requires_grad=True,
        )
        torch.nn.init.normal_(
            cls.w1, mean=0.0, std=cls.weight_scale / np.sqrt(cls.nb_inputs)
        )

        cls.w2 = torch.empty(
            (cls.nb_hidden, cls.nb_hidden[1]),
            device=cls.device,
            dtype=cls.dtype,
            requires_grad=True,
        )
        torch.nn.init.normal_(
            cls.w2, mean=0.0, std=cls.weight_scale / np.sqrt(cls.nb_hidden[0])
        )

        cls.w3 = torch.empty(
            (cls.nb_hidden[1], cls.nb_outputs),
            device=cls.device,
            dtype=cls.dtype,
            requires_grad=True,
        )
        torch.nn.init.normal_(
            cls.w3, mean=0.0, std=cls.weight_scale / np.sqrt(cls.nb_hidden[1])
        )

        cls.v1 = torch.empty(
            (cls.nb_hidden[0], cls.nb_hidden[0]),
            device=cls.device,
            dtype=cls.dtype,
            requires_grad=True,
        )
        torch.nn.init.normal_(
            cls.v1, mean=0.0, std=cls.weight_scale / np.sqrt(cls.nb_hidden)
        )

        cls.v2 = torch.empty(
            (cls.nb_hidden[1], cls.nb_hidden[1]),
            device=cls.device,
            dtype=cls.dtype,
            requires_grad=True,
        )
        torch.nn.init.normal_(
            cls.v2, mean=0.0, std=cls.weight_scale / np.sqrt(cls.nb_hidden[1])
        )

    @classmethod
    def set_device(cls):
        if torch.cuda.is_available():
            cls.device = torch.device("cuda")
        else:
            cls.device = torch.device("cpu")

    @classmethod
    def set_shd_dataset(cls):
        # Here we load the Dataset
        cache_dir = os.path.expanduser("./data")
        cache_subdir = "hdspikes"
        get_shd_dataset(cache_dir, cache_subdir)

        train_file = h5py.File(
            os.path.join(cache_dir, cache_subdir, "shd_train.h5"), "r"
        )
        test_file = h5py.File(os.path.join(cache_dir, cache_subdir, "shd_test.h5"), "r")

        cls.x_train = train_file["spikes"]
        cls.y_train = train_file["labels"]
        cls.x_test = test_file["spikes"]
        cls.y_test = test_file["labels"]

    @classmethod
    def sparse_data_generator_from_hdf5_spikes(
        cls, X, y, batch_size, nb_steps, nb_units, max_time, shuffle=True
    ):
        """This generator takes a spike dataset and generates spiking network input as sparse tensors.

        Args:
            X: The data ( sample x event x 2 ) the last dim holds (time,neuron) tuples
            y: The labels
        """

        labels_ = np.array(y, dtype=np.int64)
        number_of_batches = len(labels_) // batch_size
        sample_index = np.arange(len(labels_))

        # compute discrete firing times
        firing_times = X["times"]
        units_fired = X["units"]

        time_bins = np.linspace(0, max_time, num=nb_steps)

        if shuffle:
            np.random.shuffle(sample_index)

        counter = 0
        while counter < number_of_batches:
            batch_index = sample_index[
                batch_size * counter : batch_size * (counter + 1)
            ]

            coo = [[] for i in range(3)]
            for bc, idx in enumerate(batch_index):
                times = np.digitize(firing_times[idx], time_bins)
                units = units_fired[idx]
                batch = [bc for _ in range(len(times))]

                coo[0].extend(batch)
                coo[1].extend(times)
                coo[2].extend(units)

            i = torch.LongTensor(coo).to(cls.device)
            v = torch.FloatTensor(np.ones(len(coo[0]))).to(cls.device)

            X_batch = torch.sparse_coo_tensor(
                i, v, torch.Size([batch_size, nb_steps, nb_units])
            ).to(cls.device)
            y_batch = torch.tensor(labels_[batch_index], device=cls.device)

            yield X_batch.to(device=cls.device), y_batch.to(device=cls.device)

            counter += 1

    @classmethod
    def init_once(cls):
        cls.set_device()
        cls.set_shd_dataset()
        logging.info("init_once complete.")

    @classmethod
    def trainer_init(cls):
        if len(cls.nb_hidden) == 1:
            cls.set_weights_1()
        else:
            cls.set_weights_2()

        cls.set_spike_fn()

        logging.info("trainer_init complete.")

    @classmethod
    def save_log_to_csv(
        cls,
        filename,
        nb_hidden,
        nb_steps,
        surrogate_gradient_scale,
        nb_epochs,
        lr,
        loss_hist,
        running_time,
    ):
        """
        Train 완료 후 로그를 CSV 파일에 저장하는 함수.
        파일이 이미 존재하면 덮어쓰지 않고 내용을 추가하여 한 줄씩 기록.
        """
        # CSV 파일이 없는 경우, 헤더를 포함한 새 파일 생성
        file_exists = os.path.exists(filename)

        with open(filename, mode="a", newline="") as file:  # 'a' 모드로 파일에 내용 추가
            writer = csv.writer(file)

            # 파일이 처음 생성된 경우 헤더 작성
            if not file_exists:
                writer.writerow(
                    [
                        "hidden_layer_node",
                        "nsteps",
                        "surrogate_scale",
                        "epochs",
                        "learning_rate",
                        "loss",
                        "running_time",
                    ]
                )

            # 학습 결과를 한 줄로 기록
            writer.writerow(
                [
                    nb_hidden,
                    nb_steps,
                    surrogate_gradient_scale,
                    nb_epochs,
                    lr,
                    loss_hist[-1],
                    f"{running_time:.2f} seconds",
                ]
            )

    @classmethod
    def run_snn_2(cls, inputs):
        # 첫 번째 은닉층 변수
        syn1 = torch.zeros(
            (cls.batch_size, cls.nb_hidden[0]), device=cls.device, dtype=cls.dtype
        )
        mem1 = torch.zeros(
            (cls.batch_size, cls.nb_hidden[0]), device=cls.device, dtype=cls.dtype
        )

        # 두 번째 은닉층 변수
        syn2 = torch.zeros(
            (cls.batch_size, cls.nb_hidden[1]), device=cls.device, dtype=cls.dtype
        )
        mem2 = torch.zeros(
            (cls.batch_size, cls.nb_hidden[1]), device=cls.device, dtype=cls.dtype
        )

        mem_rec1 = []
        spk_rec1 = []

        mem_rec2 = []
        spk_rec2 = []

        # 첫 번째 은닉층의 활동
        out1 = torch.zeros(
            (cls.batch_size, cls.nb_hidden[0]), device=cls.device, dtype=cls.dtype
        )
        h1_from_input = torch.einsum("abc,cd->abd", (inputs, cls.w1))

        for t in range(cls.nb_steps):
            h1 = h1_from_input[:, t] + torch.einsum("ab,bc->ac", (out1, cls.v1))
            mthr1 = mem1 - 1.0
            out1 = cls.spike_fn(mthr1)
            rst1 = out1.detach()  # 리셋에서는 역전파 제외

            new_syn1 = cls.alpha * syn1 + h1
            new_mem1 = (cls.beta * mem1 + syn1) * (1.0 - rst1)

            mem_rec1.append(mem1)
            spk_rec1.append(out1)

            mem1 = new_mem1
            syn1 = new_syn1

        mem_rec1 = torch.stack(mem_rec1, dim=1)
        spk_rec1 = torch.stack(spk_rec1, dim=1)

        # 두 번째 은닉층의 활동
        out2 = torch.zeros(
            (cls.batch_size, cls.nb_hidden[1]), device=cls.device, dtype=cls.dtype
        )
        h2_from_hidden1 = torch.einsum("abc,cd->abd", (spk_rec1, cls.w2))

        for t in range(cls.nb_steps):
            h2 = h2_from_hidden1[:, t] + torch.einsum("ab,bc->ac", (out2, cls.v2))
            mthr2 = mem2 - 1.0
            out2 = cls.spike_fn(mthr2)
            rst2 = out2.detach()

            new_syn2 = cls.alpha * syn2 + h2
            new_mem2 = (cls.beta * mem2 + syn2) * (1.0 - rst2)

            mem_rec2.append(mem2)
            spk_rec2.append(out2)

            mem2 = new_mem2
            syn2 = new_syn2

        mem_rec2 = torch.stack(mem_rec2, dim=1)
        spk_rec2 = torch.stack(spk_rec2, dim=1)

        # 출력층
        h3 = torch.einsum("abc,cd->abd", (spk_rec2, cls.w3))
        flt = torch.zeros(
            (cls.batch_size, cls.nb_outputs), device=cls.device, dtype=cls.dtype
        )
        out = torch.zeros(
            (cls.batch_size, cls.nb_outputs), device=cls.device, dtype=cls.dtype
        )
        out_rec = [out]

        for t in range(cls.nb_steps):
            new_flt = cls.alpha * flt + h3[:, t]
            new_out = cls.beta * out + flt

            flt = new_flt
            out = new_out

            out_rec.append(out)

        out_rec = torch.stack(out_rec, dim=1)
        other_recs = [mem_rec1, spk_rec1, mem_rec2, spk_rec2]

        return out_rec, other_recs

    @classmethod
    def train_2(cls, x_data, y_data, lr=1e-3, nb_epochs=10):
        # 두 번째 은닉층 관련 가중치 (w3, v2)를 추가
        params = [cls.w1, cls.w2, cls.w3, cls.v1, cls.v2]
        optimizer = torch.optim.Adamax(params, lr=lr, betas=(0.9, 0.999))

        log_softmax_fn = nn.LogSoftmax(dim=1)
        loss_fn = nn.NLLLoss()

        start = time.time()
        loss_hist = []
        for e in range(nb_epochs):
            local_loss = []
            for x_local, y_local in cls.sparse_data_generator_from_hdf5_spikes(
                x_data,
                y_data,
                cls.batch_size,
                cls.nb_steps,
                cls.nb_inputs,
                cls.max_time,
            ):
                output, recs = cls.run_snn_2(x_local.to_dense())

                mem1_rec, spk1_rec, mem2_rec, spk2_rec = recs

                m, _ = torch.max(output, 1)
                log_p_y = log_softmax_fn(m)

                # 첫 번째 은닉층의 스파이크 기록을 사용한 정규화 손실 계산
                reg_loss = 2e-6 * torch.sum(spk1_rec)
                reg_loss += 2e-6 * torch.mean(
                    torch.sum(torch.sum(spk1_rec, dim=0), dim=0) ** 2
                )

                # 두 번째 은닉층 스파이크 기록도 정규화 손실에 포함할 수 있음
                reg_loss += 2e-6 * torch.sum(spk2_rec)
                reg_loss += 2e-6 * torch.mean(
                    torch.sum(torch.sum(spk2_rec, dim=0), dim=0) ** 2
                )

                loss_val = loss_fn(log_p_y, y_local) + reg_loss

                optimizer.zero_grad()
                loss_val.backward()
                optimizer.step()
                local_loss.append(loss_val.item())
            mean_loss = np.mean(local_loss)
            loss_hist.append(mean_loss)
            logging.info(f"Epoch: {e+1} loss={mean_loss}")

        end = time.time()

        runningtime = end - start

        return loss_hist, runningtime

    @classmethod
    def trainer(cls, nb_hidden, nb_steps, surrogate_gradient_scale, nb_epochs, lr):
        cls.nb_hidden = nb_hidden
        cls.nb_steps = nb_steps
        cls.surrogate_gradient_scale = surrogate_gradient_scale

        if len(nb_hidden) == 1:
            trainer = cls.train_1
        else:
            trainer = cls.train_2

        cls.trainer_init()
        logging.info("Train start")
        logging.info(f"hidden layer node: {nb_hidden}")
        logging.info(f"steps: {nb_steps}")
        logging.info(f"surrogate scale: {surrogate_gradient_scale}")
        logging.info(f"epochs: {nb_epochs}")
        logging.info(f"lr : {lr}")
        loss_hist, running_time = trainer(
            cls.x_train, cls.y_train, lr=lr, nb_epochs=nb_epochs
        )
        logging.info("Train end.")
        logging.info(f"loss: {loss_hist}")
        logging.info(f"running time: {running_time}")

        cls.save_log_to_csv(
            "training_log.csv",
            nb_hidden,
            nb_steps,
            surrogate_gradient_scale,
            nb_epochs,
            lr,
            loss_hist,
            running_time,
        )

    @classmethod
    def grid_search(cls, params):
        cls.init_once()
        total_count = (
            len(params["nb_hidden"])
            * len(params["nb_steps"])
            * len(params["scale"])
            * len(params["nb_epochs"])
            * len(params["lr"])
        )
        count = 0
        for hidden in params["nb_hidden"]:
            for steps in params["nb_steps"]:
                for scale in params["scale"]:
                    for epoch in params["nb_epochs"]:
                        for lr in params["lr"]:
                            count += 1
                            logging.info(f"Training Model count: {count}/{total_count}")
                            cls.trainer_init()
                            cls.trainer(
                                nb_hidden=hidden,
                                nb_steps=steps,
                                surrogate_gradient_scale=scale,
                                nb_epochs=epoch,
                                lr=lr,
                            )


####### Main #######
seed = 1004
np.random.seed(seed)
torch.manual_seed(seed)

params = {
    "nb_hidden": [
        [50],
        [100],
        [200],
        [50, 50],
        [100, 100],
        [200, 200],
    ],
    "nb_steps": [50, 100, 150],
    "scale": [100],
    "nb_epochs": [100, 200, 300],
    "lr": [1e-2, 1e-3, 1e-4],
}

VoiceData_Tutorial_4.grid_search(params)
