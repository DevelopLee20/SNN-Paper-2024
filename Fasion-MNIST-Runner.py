import os
import numpy as np
import logging
import torch
import torch.nn as nn
import torchvision
import csv
import time

from SurrogateGradient import SurrogateGradientSpike

logging.basicConfig(
    level=logging.DEBUG,  # 로그 레벨 설정
    format="%(asctime)s - %(levelname)s - %(message)s",  # 로그 메시지 형식
    handlers=[
        logging.FileHandler("Fasion-MNIST-Runner.log"),  # 로그를 기록할 파일 설정
        logging.StreamHandler(),  # 콘솔에 로그 출력
    ],
)


class FasionMNISTRunner:
    # Hyper Parameters
    nb_hidden = 100
    nb_steps = 100
    batch_size = 256
    regularizer = 1e-5
    lr = 1e-3
    nb_epochs = 30

    # Constrant Parameters
    nb_inputs = 28 * 28
    nb_outputs = 10
    time_step = 1e-3
    tau_mem = 10e-3
    tau_syn = 5e-3
    alpha = float(np.exp(-time_step / tau_syn))
    beta = float(np.exp(-time_step / tau_mem))
    log_softmax_fn = nn.LogSoftmax(dim=1)
    loss_fn = nn.NLLLoss()

    # Global Values
    dtype = torch.float

    # Set Once
    device = None
    x_train = None
    y_train = None
    x_test = None
    y_test = None

    # Set one train
    w1 = None
    w2 = None
    spike_fn = None

    @classmethod
    def __init__(cls):
        cls.set_device()
        cls.set_datasets()

    @classmethod
    def set_params(cls, params: dict):
        """
        은닉층 노드
        시간 스텝
        서로게이트 스케일
        에포치
        학습률
        정규화 손실 상수
        """
        cls.nb_hidden = params["hidden_node"]
        cls.nb_steps = params["steps"]
        cls.set_spike_fn(scale=params["scale"])
        cls.nb_epochs = params["epochs"]
        cls.lr = params["lr"]
        cls.regularizer = params["regularizer"]

        logging.info("Parameters Option============")
        logging.info(f"hidden node: {params['hidden_node']}")
        logging.info(f"time step: {params['steps']}")
        logging.info(f"surrogate scale: {params['scale']}")
        logging.info(f"epochs: {params['epochs']}")
        logging.info(f"learning rate: {params['lr']}")
        logging.info(f"regularizer loss: {params['regularizer']}")

    @classmethod
    def runner(cls, params: dict):
        cls.set_params(params)
        cls.set_weights()

        start = time.time()
        loss_hist = cls.train(cls.x_train, cls.y_train)
        end = time.time()

        cls.save_log_to_csv(
            surrogate_gradient_scale=params["scale"],
            loss=loss_hist[-1],
            running_time=round(end - start, 2),
            train_acc=round(
                cls.compute_classification_accuracy(cls.x_train, cls.y_train) * 100, 2
            ),
            test_acc=round(
                cls.compute_classification_accuracy(cls.x_test, cls.y_test) * 100, 2
            ),
        )

    @classmethod
    def set_device(cls):
        if torch.cuda.is_available():
            cls.device = torch.device("cuda")
        else:
            cls.device = torch.device("cpu")

    @classmethod
    def set_datasets(cls):
        root = os.path.expanduser("./datasets/fasion-mnist")

        train_dataset = torchvision.datasets.FashionMNIST(
            root, train=True, transform=None, target_transform=None, download=True
        )
        test_dataset = torchvision.datasets.FashionMNIST(
            root, train=False, transform=None, target_transform=None, download=True
        )

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
    def current2firing_time(cls, x, tau=20, thr=0.2, tmax=1.0, epsilon=1e-7):
        idx = x < thr
        x = np.clip(x, thr + epsilon, 1e9)
        T = tau * np.log(x / (x - thr))
        T[idx] = tmax
        return T

    @classmethod
    def sparse_data_generator(cls, X, y, shuffle=True):
        labels_ = np.array(y, dtype=np.int64)
        number_of_batches = len(X) // cls.batch_size
        sample_index = np.arange(len(X))

        tau_eff = 20e-3 / cls.time_step
        firing_times = np.array(
            cls.current2firing_time(X, tau=tau_eff, tmax=cls.nb_steps), dtype=np.int64
        )
        unit_numbers = np.arange(cls.nb_inputs)

        if shuffle:
            np.random.shuffle(sample_index)

        counter = 0
        while counter < number_of_batches:
            batch_index = sample_index[
                cls.batch_size * counter : cls.batch_size * (counter + 1)
            ]

            coo = [[] for i in range(3)]
            for bc, idx in enumerate(batch_index):
                c = firing_times[idx] < cls.nb_steps
                times, units = firing_times[idx][c], unit_numbers[c]

                batch = [bc for _ in range(len(times))]
                coo[0].extend(batch)
                coo[1].extend(times)
                coo[2].extend(units)

            i = torch.LongTensor(coo).to(cls.device)
            v = torch.FloatTensor(np.ones(len(coo[0]))).to(cls.device)

            X_batch = torch.sparse_coo_tensor(
                i, v, torch.Size([cls.batch_size, cls.nb_steps, cls.nb_inputs])
            ).to(cls.device)
            y_batch = torch.tensor(labels_[batch_index], device=cls.device)

            yield X_batch.to(device=cls.device), y_batch.to(device=cls.device)

            counter += 1

    @classmethod
    def set_weights(cls):
        weight_scale = 0.2

        cls.w1 = torch.empty(
            (cls.nb_inputs, cls.nb_hidden),
            device=cls.device,
            dtype=cls.dtype,
            requires_grad=True,
        )
        torch.nn.init.normal_(
            cls.w1, mean=0.0, std=weight_scale / np.sqrt(cls.nb_inputs)
        )

        cls.w2 = torch.empty(
            (cls.nb_hidden, cls.nb_outputs),
            device=cls.device,
            dtype=cls.dtype,
            requires_grad=True,
        )
        torch.nn.init.normal_(
            cls.w2, mean=0.0, std=weight_scale / np.sqrt(cls.nb_hidden)
        )

    @classmethod
    def set_spike_fn(cls, scale):
        cls.spike_fn = SurrogateGradientSpike(scale=scale).apply

    @classmethod
    def run_snn(cls, inputs):
        h1 = torch.einsum("abc,cd->abd", (inputs, cls.w1))
        syn = torch.zeros(
            (cls.batch_size, cls.nb_hidden), device=cls.device, dtype=cls.dtype
        )
        mem = torch.zeros(
            (cls.batch_size, cls.nb_hidden), device=cls.device, dtype=cls.dtype
        )

        mem_rec = []
        spk_rec = []

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
    def train(cls, x_data, y_data):
        params = [cls.w1, cls.w2]
        optimizer = torch.optim.Adamax(params, lr=cls.lr, betas=(0.9, 0.999))

        loss_hist = []
        for e in range(cls.nb_epochs):
            local_loss = []
            for x_local, y_local in cls.sparse_data_generator(x_data, y_data):
                output, recs = cls.run_snn(x_local.to_dense())
                _, spks = recs
                m, _ = torch.max(output, 1)
                log_p_y = cls.log_softmax_fn(m)

                # Here we set up our regularizer loss
                # The strength paramters here are merely a guess and there should be ample room for improvement by
                # tuning these paramters.
                reg_loss = cls.regularizer * torch.sum(
                    spks
                )  # L1 loss on total number of spikes
                reg_loss += cls.regularizer * torch.mean(
                    torch.sum(torch.sum(spks, dim=0), dim=0) ** 2
                )  # L2 loss on spikes per neuron

                # Here we combine supervised loss and the regularizer
                loss_val = cls.loss_fn(log_p_y, y_local) + reg_loss

                optimizer.zero_grad()
                loss_val.backward()
                optimizer.step()
                local_loss.append(loss_val.item())
            mean_loss = np.mean(local_loss)
            logging.info(f"Epoch {e+1}: loss={mean_loss:.5f}")
            loss_hist.append(mean_loss)

        return loss_hist

    @classmethod
    def compute_classification_accuracy(cls, x_data, y_data):
        accs = []
        for x_local, y_local in cls.sparse_data_generator(
            x_data, y_data, cls.batch_size, cls.nb_steps, cls.nb_inputs, shuffle=False
        ):
            output, _ = cls.run_snn(x_local.to_dense())
            m, _ = torch.max(output, 1)
            _, am = torch.max(m, 1)
            tmp = np.mean((y_local == am).detach().cpu().numpy())
            accs.append(tmp)

        return np.mean(accs)

    @classmethod
    def save_log_to_csv(
        cls,
        surrogate_gradient_scale,
        loss,
        running_time,
        train_acc,
        test_acc,
    ):
        """
        Train 완료 후 로그를 CSV 파일에 저장하는 함수.
        파일이 이미 존재하면 덮어쓰지 않고 내용을 추가하여 한 줄씩 기록.
        """
        # CSV 파일이 없는 경우, 헤더를 포함한 새 파일 생성
        file_exists = os.path.exists("train_result_Fasion_MNIST.csv")

        with open(
            "train_result_Fasion_MNIST.csv", mode="a", newline=""
        ) as file:  # 'a' 모드로 파일에 내용 추가
            writer = csv.writer(file)

            # 파일이 처음 생성된 경우 헤더 작성
            if not file_exists:
                writer.writerow(
                    [
                        "hidden node",
                        "steps",
                        "surrogate scale",
                        "epochs",
                        "learning rate",
                        "loss",
                        "running time",
                        "train accuracy",
                        "test_accuracy",
                    ]
                )

            # 학습 결과를 한 줄로 기록
            writer.writerow(
                [
                    cls.nb_hidden,
                    cls.nb_steps,
                    surrogate_gradient_scale,
                    cls.nb_epochs,
                    cls.lr,
                    loss,
                    running_time,
                    train_acc,
                    test_acc,
                ]
            )

        print(f"model loss: {loss}")


if __name__ == "__main__":
    seed = 1004
    np.random.seed(seed)
    torch.manual_seed(seed)

    params = {
        "hidden_node": [75, 100, 125],
        "steps": [75, 100, 125],
        "scale": [95, 100, 105],
        "epochs": [30, 50, 100],
        "lr": [1e-4, 2e-4, 3e-4],
        "regularizer": [1e-4, 1e-5, 1e-6],
    }
    runner = FasionMNISTRunner()

    total_count = 729
    count = 0
    for h_node in params["hidden_node"]:
        for steps in params["steps"]:
            for scale in params["scale"]:
                for epoch in params["epochs"]:
                    for lr in params["lr"]:
                        for regular in params["regularizer"]:
                            param = {
                                "hidden_node": h_node,
                                "steps": steps,
                                "scale": scale,
                                "epochs": epoch,
                                "lr": lr,
                                "regularizer": regular,
                            }
                            count += 1
                            print(f"{count}/{total_count} is started.")
                            runner.runner(param)
                            print(f"{count}/{total_count} is end.")
