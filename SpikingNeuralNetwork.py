import torch
import numpy as np
import torchvision
import os
import logging
import pickle
import time

from SurrogateGradient import SurrogateGradientSpike

# 기본 설정
logging.basicConfig(
    level=logging.DEBUG,  # 로그 레벨 설정
    format="%(asctime)s - %(levelname)s - %(message)s",  # 로그 메시지 형식
    handlers=[
        logging.FileHandler("Tutorial3.log"),  # 로그를 기록할 파일 설정
        logging.StreamHandler(),  # 콘솔에 로그 출력
    ],
)


class Tutorial3_SNN_Runner:
    nb_inputs = 28 * 28  # 입력크기
    nb_hidden = [100]  # 은닉층 param
    nb_outputs = 10  # 출력층
    time_step = 1e-3  # 단위시간
    nb_steps = 100  # 시간스텝
    batch_size = 256  # 배치크기
    nb_epochs = 10  # 반복횟수 param
    dtype = torch.float
    tau_mem = 10e-3
    tau_syn = 5e-3
    lr = 1e-3  # 학습률 param
    alpha = float(np.exp(-time_step / tau_syn))  # He 초기값
    beta = float(np.exp(-time_step / tau_mem))  # He 초기값
    weight_scale = 0.2  # He 초기값
    weight_list = None
    w1 = None
    w2 = None
    device = None
    spike_fn = SurrogateGradientSpike.apply
    x_train = None
    y_train = None
    x_test = None
    y_test = None

    @classmethod
    def _current2firing_time(
        cls,
        x: any,
        tau: int = 20,
        thr: float = 0.2,
        tmax: float = 1.0,
        epsilon: float = 1e-7,
    ):
        """스파이크가 발생한 시간을 기준으로 스파이크의 강도를 계산하는 함수
        [Params]
        x       : 현재 값
        tau     : 뉴런의 막전위가 증가되는 최소 속도 보장
        thr     : 스파이크 발생 임계값
        tmax    : 스파이크 발생하지 않을 때 해당 값으로 지정
        epsilon : Division Error 발생 방지를 위해 사용

        [Return]
        T       : x에 대한 스파이크 발생 시간
        """
        idx = x < thr
        x = np.clip(x, thr + epsilon, 1e9)
        T = tau * np.log(x / (x - thr))
        T[idx] = tmax

        return T

    @classmethod
    def sparse_data_generator(cls, X, y, batch_size, nb_steps, nb_units, shuffle=True):
        """데이터 셋을 SNN용 희소행렬로 변환 후 반환하는 제너레이터
        [Params]
        X            : 입력 데이터
        y            : 라벨 데이터
        batch_size   : 배치 크기
        nb_steps     : 시간 단위 개수
        nb_units     : 은닉층 뉴런 개수
        shuffle      : 데이터 셔플 여부
        """

        labels_ = np.array(y, dtype=int)
        number_of_batches = len(X) // batch_size
        sample_index = np.arange(len(X))

        # compute discrete firing times
        tau_eff = 20e-3 / cls.time_step
        firing_times = np.array(
            cls._current2firing_time(X, tau=tau_eff, tmax=nb_steps), dtype=int
        )
        unit_numbers = np.arange(nb_units)

        if shuffle:
            np.random.shuffle(sample_index)

        counter = 0
        while counter < number_of_batches:
            batch_index = sample_index[
                batch_size * counter : batch_size * (counter + 1)
            ]

            coo = [[] for i in range(3)]
            for bc, idx in enumerate(batch_index):
                c = firing_times[idx] < nb_steps
                times, units = firing_times[idx][c], unit_numbers[c]

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
    def set_pytorch_device(cls):
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

        cls.device = device

    @classmethod
    def set_download_FashionMNIST(cls):
        root = "./data/fasionMNIST"

        download = True
        if os.path.exists(os.path.join(root, "FashionMNIST")):
            download = False

        train_dataset = torchvision.datasets.FashionMNIST(
            root, train=True, transform=None, target_transform=None, download=download
        )
        test_dataset = torchvision.datasets.FashionMNIST(
            root, train=False, transform=None, target_transform=None, download=download
        )

        # 정규화
        x_train = np.array(train_dataset.data, dtype=float)
        x_train = x_train.reshape(x_train.shape[0], -1) / 255
        x_test = np.array(test_dataset.data, dtype=float)
        x_test = x_test.reshape(x_test.shape[0], -1) / 255

        y_train = np.array(train_dataset.targets, dtype=int)
        y_test = np.array(test_dataset.targets, dtype=int)

        cls.x_train = x_train
        cls.y_train = y_train
        cls.x_test = x_test
        cls.y_test = y_test

    @classmethod
    def set_layers_weight_list(cls):
        w1 = torch.empty(
            (cls.nb_inputs, cls.nb_hidden[0]),
            device=cls.device,
            dtype=cls.dtype,
            requires_grad=True,
        )
        torch.nn.init.normal_(
            w1, mean=0.0, std=cls.weight_scale / np.sqrt(cls.nb_inputs)
        )
        cls.w1 = w1

        w2 = torch.empty(
            (cls.nb_hidden[-1], cls.nb_outputs),
            device=cls.device,
            dtype=cls.dtype,
            requires_grad=True,
        )
        torch.nn.init.normal_(
            w2, mean=0.0, std=cls.weight_scale / np.sqrt(cls.nb_hidden[-1])
        )
        cls.w2 = w2

        logging.info("weight init done.")

    @classmethod
    def run_snn_h1(cls, inputs):
        h1 = torch.einsum("abc,cd->abd", (inputs, cls.w1))
        syn = torch.zeros(
            (cls.batch_size, cls.nb_hidden[0]), device=cls.device, dtype=cls.dtype
        )
        mem = torch.zeros(
            (cls.batch_size, cls.nb_hidden[0]), device=cls.device, dtype=cls.dtype
        )

        mem_rec = []
        spk_rec = []

        # Compute hidden layer activity
        for t in range(cls.nb_steps):
            mthr = mem - 1.0
            out = cls.spike_fn(mthr)
            rst = out.detach()  # We do not want to backprop through the reset

            new_syn = cls.alpha * syn + h1[:, t]
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
    def run_snn_h2(cls, inputs):
        pass

    @classmethod
    def run_snn_h3(cls, inputs):
        pass

    @classmethod
    def train(cls, x_data, y_data, lr=1e-3, nb_epochs=10):
        if len(cls.nb_hidden) == 1:
            snn_runner = cls.run_snn_h1
            params = [cls.w1, cls.w2]

        if len(cls.nb_hidden) == 2:
            snn_runner = cls.run_snn_h2
            params = [cls.w1, cls.weight_list[0], cls.w2]

        if len(cls.nb_hidden) == 3:
            snn_runner = cls.run_snn_h3
            params = [cls.w1, cls.weight_list[0], cls.weight_list[1], cls.w2]

        optimizer = torch.optim.Adamax(params, lr=lr, betas=(0.9, 0.999))

        log_softmax_fn = torch.nn.LogSoftmax(dim=1)
        loss_fn = torch.nn.NLLLoss()

        loss_hist = []
        for e in range(nb_epochs):
            local_loss = []
            for x_local, y_local in cls.sparse_data_generator(
                x_data, y_data, cls.batch_size, cls.nb_steps, cls.nb_inputs
            ):
                output, recs = snn_runner(x_local.to_dense())
                _, spks = recs
                m, _ = torch.max(output, 1)
                log_p_y = log_softmax_fn(m)

                # Here we set up our regularizer loss
                # The strength paramters here are merely a guess and there should be ample room for improvement by
                # tuning these paramters.
                reg_loss = 1e-5 * torch.sum(spks)  # L1 loss on total number of spikes
                reg_loss += 1e-5 * torch.mean(
                    torch.sum(torch.sum(spks, dim=0), dim=0) ** 2
                )  # L2 loss on spikes per neuron

                # Here we combine supervised loss and the regularizer
                loss_val = loss_fn(log_p_y, y_local) + reg_loss

                optimizer.zero_grad()
                loss_val.backward()
                optimizer.step()
                local_loss.append(loss_val.item())
            mean_loss = np.mean(local_loss)
            logging.info(f"Epoch: {e+1} loss={mean_loss}")
            loss_hist.append(mean_loss)

        return loss_hist

    @classmethod
    def compute_classification_accuracy(cls, x_data, y_data):
        """Computes classification accuracy on supplied data in batches."""
        accs = []
        for x_local, y_local in cls.sparse_data_generator(
            x_data, y_data, cls.batch_size, cls.nb_steps, cls.nb_inputs, shuffle=False
        ):
            output, _ = cls.run_snn(x_local.to_dense())
            m, _ = torch.max(output, 1)  # max over time
            _, am = torch.max(m, 1)  # argmax over output units
            tmp = np.mean((y_local == am).detach().cpu().numpy())  # compare to labels
            accs.append(tmp)
        return np.mean(accs)

    @classmethod
    def trainer(cls, params_dict: dict):
        logging.info("Trainer Start")
        cls.set_download_FashionMNIST()
        logging.info(f"Data Size: {cls.x_train.shape} {cls.x_test.shape}")
        cls.set_pytorch_device()
        logging.info(f"Hidden count: {len(params_dict['nb_hidden'])}")
        logging.info(f"Epochs count: {len(params_dict['nb_epochs'])}")
        logging.info(f"lr count: {len(params_dict['lr'])}")
        total_model = (
            len(params_dict["nb_hidden"])
            * len(params_dict["nb_epochs"])
            * len(params_dict["lr"])
        )
        logging.info(f"total model: {total_model}")

        count = 0
        for nb_hidden in params_dict["nb_hidden"]:
            for epochs in params_dict["nb_epochs"]:
                for lr in params_dict["lr"]:
                    count += 1
                    logging.info(
                        f"[{count}/{total_model}] {nb_hidden}_{epochs}_{lr} model running."
                    )
                    logging.info(f"hidden Layer: {nb_hidden}")
                    logging.info(f"Epochs: {epochs}")
                    logging.info(f"lr: {lr}")

                    cls.nb_hidden = nb_hidden
                    cls.nb_epochs = epochs
                    cls.lr = lr
                    cls.set_layers_weight_list()

                    start = time.time()
                    result = cls.train(cls.x_train, cls.y_train)
                    end = time.time()

                    logging.info(f"Model End, Time: {end-start:.2f} sec")

                    with open(
                        f"result/{result[-1]}_{nb_hidden}_{epochs}_{lr}.pkl", "wb"
                    ) as file:
                        pickle.dump(result, file=file)

                    logging.info(
                        f"save pickle: result/{result[-1]}_{nb_hidden}_{epochs}_{lr}.pkl"
                    )


if __name__ == "__main__":
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
            [50, 50, 50],
            [100, 100, 100],
            [200, 200, 200],
        ],
        "nb_epochs": [5, 10, 15, 50, 100],
        "lr": [1e-1, 1e-2, 1e-3, 1e-4, 1e-5],
    }

    Tutorial3_SNN_Runner.trainer(params_dict=params)
