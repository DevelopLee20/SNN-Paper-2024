import os
import logging
import numpy as np
import torch
import torch.nn as nn
import time
import csv
import pickle
import gzip
from torch.utils.data import TensorDataset, DataLoader

from SurrogateGradient import SurrogateGradientSpike

logging.basicConfig(
    level=logging.DEBUG,  # 로그 레벨 설정
    format="%(asctime)s - %(levelname)s - %(message)s",  # 로그 메시지 형식
    handlers=[
        logging.FileHandler("logs/Tactile-Braille-Letters-Runner.log"),  # 로그를 기록할 파일 설정
        logging.StreamHandler(),  # 콘솔에 로그 출력
    ],
)


class TactileBrailleLettersRunner:
    # Hyper Parameters
    nb_hidden = 200
    batch_size = 100
    lr = 1e-2
    nb_epochs = 100
    # scale = 20

    # Constrant Parameters
    log_softmax_fn = nn.LogSoftmax(dim=1)
    loss_fn = nn.NLLLoss()

    # Global Values
    dtype = torch.float
    enc_fan_out = 32  # Num of spiking neurons used to encode each channel
    nb_upsample = 2  # 최소 데이터를 배수로 증가
    nb_steps = None
    data_steps = None

    # Set Once
    device = None
    ds_train = None
    ds_test = None
    nb_inputs = None
    nb_outputs = None
    alpha = None
    beta = None

    # Set one train
    enc_gain = None
    enc_bias = None
    w1 = None
    w2 = None
    v1 = None
    spike_fn = None

    @classmethod
    def __init__(cls):
        cls.set_device()
        cls.set_datasets_and_network_params()

    @classmethod
    def set_spike_fn(cls, scale):
        cls.spike_fn = SurrogateGradientSpike(scale=scale).apply

    @classmethod
    def set_params(cls, params: dict):
        """
        은닉층 노드
        업샘플링 배수
        서로게이트 스케일
        에포치
        학습률
        정규화 손실 상수
        """
        cls.nb_hidden = params["hidden_node"]

        cls.nb_upsample = params["upsample"]
        cls.nb_steps = cls.nb_upsample * cls.data_steps

        cls.set_spike_fn(scale=params["scale"])
        cls.nb_epochs = params["epochs"]
        cls.lr = params["lr"]
        cls.regularizer = params["regularizer"]

        logging.info("Parameters Option============")
        logging.info(f"hidden node: {params['hidden_node']}")
        logging.info(f"upsample: {params['upsample']}")
        logging.info(f"step: {cls.nb_steps}")
        logging.info(f"surrogate scale: {params['scale']}")
        logging.info(f"epochs: {params['epochs']}")
        logging.info(f"learning rate: {params['lr']}")
        logging.info(f"regularizer loss: {params['regularizer']}")

    @classmethod
    def set_device(cls):
        if torch.cuda.is_available():
            cls.device = torch.device("cuda")
        else:
            cls.device = torch.device("cpu")

    @classmethod
    def set_datasets_and_network_params(cls):
        file_name = "tutorial5_braille_spiking_data.pkl.gz"
        with gzip.open(file_name, "rb") as infile:
            data_dict = pickle.load(infile)

        letter_written = [
            "Space",
            "A",
            "B",
            "C",
            "D",
            "E",
            "F",
            "G",
            "H",
            "I",
            "J",
            "K",
            "L",
            "M",
            "N",
            "O",
            "P",
            "Q",
            "R",
            "S",
            "T",
            "U",
            "V",
            "W",
            "X",
            "Y",
            "Z",
        ]

        # Extract data
        nb_repetitions = 50
        data = []
        labels = []
        for i, letter in enumerate(letter_written):
            for repetition in np.arange(nb_repetitions):
                idx = i * nb_repetitions + repetition
                dat = 1.0 - data_dict[idx]["taxel_data"][:] / 255
                data.append(dat)
                labels.append(i)

        # Crop to same length
        cls.data_steps = lett = np.min([len(d) for d in data])
        data_np = np.array([d[:lett] for d in data])
        data = torch.tensor(data_np, dtype=cls.dtype)
        labels = torch.tensor(labels, dtype=torch.long)

        # Select nonzero inputs
        nzid = [1, 2, 6, 10]
        data = data[:, :, nzid]

        # Standardize data
        rshp = data.reshape((-1, data.shape[2]))
        data = (data - rshp.mean(0)) / (rshp.std(0) + 1e-3)

        # Upsample
        def upsample(data, n=2):
            shp = data.shape
            tmp = data.reshape(shp + (1,))
            tmp = data.tile((1, 1, 1, n))
            return tmp.reshape((shp[0], n * shp[1], shp[2]))

        data = upsample(data, n=cls.nb_upsample)

        # Shuffle data
        idx = np.arange(len(data))
        np.random.shuffle(idx)
        data = data[idx]
        labels = labels[idx]

        # Peform train/test split
        a = int(0.8 * len(idx))
        x_train, x_test = data[:a], data[a:]
        y_train, y_test = labels[:a], labels[a:]

        cls.ds_train = TensorDataset(x_train, y_train)
        cls.ds_test = TensorDataset(x_test, y_test)

        nb_channels = len(nzid)

        # Network parameters
        cls.nb_inputs = nb_channels * cls.enc_fan_out
        cls.nb_outputs = len(np.unique(labels)) + 1
        time_step = (
            2e-3 / cls.nb_upsample
        )  # TODO needs to be updated to reflect the correct time scale
        cls.nb_steps = (
            cls.nb_upsample * cls.data_steps
        )  # TODO We should change this and upsample the input data

        tau_mem = 20e-3
        tau_syn = 10e-3

        cls.alpha = float(np.exp(-time_step / tau_syn))
        cls.beta = float(np.exp(-time_step / tau_mem))

    @classmethod
    def set_weights(cls):
        encoder_weight_scale = 1.0
        fwd_weight_scale = 3.0
        rec_weight_scale = 1e-2 * fwd_weight_scale

        # Encoder
        cls.enc_gain = torch.empty(
            (cls.nb_inputs,), device=cls.device, dtype=cls.dtype, requires_grad=True
        )
        cls.enc_bias = torch.empty(
            (cls.nb_inputs,), device=cls.device, dtype=cls.dtype, requires_grad=True
        )
        torch.nn.init.normal_(
            cls.enc_gain, mean=0.0, std=encoder_weight_scale
        )  # TODO update this parameter
        torch.nn.init.normal_(cls.enc_bias, mean=0.0, std=1.0)

        # Spiking network
        cls.w1 = torch.empty(
            (cls.nb_inputs, cls.nb_hidden),
            device=cls.device,
            dtype=cls.dtype,
            requires_grad=True,
        )
        torch.nn.init.normal_(
            cls.w1, mean=0.0, std=fwd_weight_scale / np.sqrt(cls.nb_inputs)
        )

        cls.w2 = torch.empty(
            (cls.nb_hidden, cls.nb_outputs),
            device=cls.device,
            dtype=cls.dtype,
            requires_grad=True,
        )
        torch.nn.init.normal_(
            cls.w2, mean=0.0, std=fwd_weight_scale / np.sqrt(cls.nb_hidden)
        )

        cls.v1 = torch.empty(
            (cls.nb_hidden, cls.nb_hidden),
            device=cls.device,
            dtype=cls.dtype,
            requires_grad=True,
        )
        torch.nn.init.normal_(
            cls.v1, mean=0.0, std=rec_weight_scale / np.sqrt(cls.nb_hidden)
        )

    @classmethod
    def run_snn(cls, inputs):
        bs = inputs.shape[0]
        enc = torch.zeros((bs, cls.nb_inputs), device=cls.device, dtype=cls.dtype)
        input_spk = torch.zeros((bs, cls.nb_inputs), device=cls.device, dtype=cls.dtype)
        syn = torch.zeros((bs, cls.nb_hidden), device=cls.device, dtype=cls.dtype)
        mem = -1e-3 * torch.ones(
            (bs, cls.nb_hidden), device=cls.device, dtype=cls.dtype
        )
        out = torch.zeros((bs, cls.nb_hidden), device=cls.device, dtype=cls.dtype)

        enc_rec = []
        mem_rec = []
        spk_rec = []

        # encoder_currents = torch.einsum("abc,c->ab", (inputs.tile((enc_fan_out,)), enc_gain))+enc_bias
        encoder_currents = cls.enc_gain * (
            inputs.tile((cls.enc_fan_out,)) + cls.enc_bias
        )
        for t in range(cls.nb_steps):
            # Compute encoder activity
            new_enc = (cls.beta * enc + (1.0 - cls.beta) * encoder_currents[:, t]) * (
                1.0 - input_spk.detach()
            )
            input_spk = cls.spike_fn(enc - 1.0)

            # Compute hidden layer activity
            h1 = input_spk.mm(cls.w1) + torch.einsum("ab,bc->ac", (out, cls.v1))
            mthr = mem - 1.0
            out = cls.spike_fn(mthr)
            rst = out.detach()  # We do not want to backprop through the reset

            new_syn = cls.alpha * syn + h1
            new_mem = (cls.beta * mem + (1.0 - cls.beta) * syn) * (1.0 - rst)

            # Here we store some state variables so we can look at them later.
            mem_rec.append(mem)
            spk_rec.append(out)
            enc_rec.append(enc)

            enc = new_enc
            mem = new_mem
            syn = new_syn

        enc_rec = torch.stack(enc_rec, dim=1)
        mem_rec = torch.stack(mem_rec, dim=1)
        spk_rec = torch.stack(spk_rec, dim=1)

        # Readout layer
        h2 = torch.einsum("abc,cd->abd", (spk_rec, cls.w2))
        flt = torch.zeros((bs, cls.nb_outputs), device=cls.device, dtype=cls.dtype)
        out = torch.zeros((bs, cls.nb_outputs), device=cls.device, dtype=cls.dtype)
        out_rec = [out]
        for t in range(cls.nb_steps):
            new_flt = cls.alpha * flt + h2[:, t]
            new_out = cls.beta * out + (1.0 - cls.beta) * flt

            flt = new_flt
            out = new_out

            out_rec.append(out)

        out_rec = torch.stack(out_rec, dim=1)
        other_recs = [enc_rec.detach(), mem_rec.detach(), spk_rec.detach()]
        return out_rec, other_recs

    @classmethod
    def train(cls, dataset):
        params = [cls.enc_gain, cls.enc_bias, cls.w1, cls.w2, cls.v1]
        optimizer = torch.optim.Adamax(params, lr=cls.lr, betas=(0.9, 0.995))

        generator = DataLoader(
            dataset, batch_size=cls.batch_size, shuffle=True, num_workers=2
        )

        loss_hist = []
        for e in range(cls.nb_epochs):
            local_loss = []
            for x_local, y_local in generator:
                x_local, y_local = x_local.to(cls.device), y_local.to(cls.device)
                output, recs = cls.run_snn(x_local)
                _, _, spks = recs
                m, _ = torch.max(output, 1)
                log_p_y = cls.log_softmax_fn(m)

                # Here we can set up our regularizer loss
                reg_loss = 1e-3 * torch.mean(
                    torch.sum(spks, 1)
                )  # e.g., L1 loss on total number of spikes
                # reg_loss = 0.0

                # Here we combine supervised loss and the regularizer
                loss_val = cls.loss_fn(log_p_y, y_local) + reg_loss

                optimizer.zero_grad()
                loss_val.backward()
                optimizer.step()
                local_loss.append(loss_val.item())

            mean_loss = np.mean(local_loss)
            loss_hist.append(mean_loss)
            logging.info(f"Epoch {e+1}: loss={mean_loss:.5f}")

        return loss_hist

    @classmethod
    def compute_classification_accuracy(cls, dataset):
        """Computes classification accuracy on supplied data in batches."""
        generator = DataLoader(
            dataset, batch_size=cls.batch_size, shuffle=False, num_workers=2
        )
        accs = []
        for x_local, y_local in generator:
            x_local, y_local = x_local.to(cls.device), y_local.to(cls.device)
            output, _ = cls.run_snn(x_local)
            m, _ = torch.max(output, 1)  # max over time
            _, am = torch.max(m, 1)  # argmax over output units
            tmp = np.mean((y_local == am).detach().cpu().numpy())  # compare to labels
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
        file_exists = os.path.exists("csv/train_result_Tactile_Braille_Letters.csv")

        with open(
            "csv/train_result_Tactile_Braille_Letters.csv", mode="a", newline=""
        ) as file:  # 'a' 모드로 파일에 내용 추가
            writer = csv.writer(file)

            # 파일이 처음 생성된 경우 헤더 작성
            if not file_exists:
                writer.writerow(
                    [
                        "hidden node",
                        "upsample",
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
                    cls.nb_upsample,
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

    @classmethod
    def runner(cls, params: dict):
        cls.set_params(params)
        cls.set_weights()

        start = time.time()
        loss_hist = cls.train(cls.ds_train)
        end = time.time()

        cls.save_log_to_csv(
            surrogate_gradient_scale=params["scale"],
            loss=loss_hist[-1],
            running_time=round(end - start, 2),
            train_acc=round(cls.compute_classification_accuracy(cls.ds_train) * 100, 2),
            test_acc=round(cls.compute_classification_accuracy(cls.ds_test) * 100, 2),
        )


if __name__ == "__main__":
    seed = 1004
    np.random.seed(seed)
    torch.manual_seed(seed)

    params = {
        "hidden_node": [200, 150, 250],
        "upsample": [2, 1, 3],
        "scale": [20.0, 25.0, 15.0],
        "epochs": [100],
        "lr": [1e-2],
        "regularizer": [1e-3, 2e-3, 5e-4],
    }
    runner = TactileBrailleLettersRunner()

    total_count = (
        len(params["hidden_node"])
        * len(params["upsample"])
        * len(params["scale"])
        * len(params["epochs"])
        * len(params["lr"])
        * len(params["regularizer"])
    )
    count = 0
    for h_node in params["hidden_node"]:
        for upsample in params["upsample"]:
            for scale in params["scale"]:
                for epoch in params["epochs"]:
                    for lr in params["lr"]:
                        for regular in params["regularizer"]:
                            param = {
                                "hidden_node": h_node,
                                "upsample": upsample,
                                "scale": scale,
                                "epochs": epoch,
                                "lr": lr,
                                "regularizer": regular,
                            }
                            count += 1
                            print(f"{count}/{total_count} is started.")
                            runner.runner(param)
                            print(f"{count}/{total_count} is end.")
