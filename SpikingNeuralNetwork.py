import torch
import numpy as np
import torchvision
import os

from SurrogateGradient import SurrogateGradientSpike


class Tutorial3_SNN_Runner:
    nb_inputs = 28 * 28  # 입력크기
    nb_hidden = [100]  # 은닉층
    nb_outputs = 10  # 출력층
    time_step = 1e-3  # 단위시간
    nb_steps = 100  # 시간스텝
    batch_size = 256  # 배치크기
    dtype = torch.float
    tau_mem = 10e-3
    tau_syn = 5e-3
    alpha = float(np.exp(-time_step / tau_syn))  # He 초기값
    beta = float(np.exp(-time_step / tau_mem))  # He 초기값
    weight_scale = 0.2  # He 초기값
    weight_list = None
    device = None
    spike_fn = SurrogateGradientSpike.apply

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

            cls.device = cls._get_pytorch_device()

            i = torch.LongTensor(coo).to(cls.device)
            v = torch.FloatTensor(np.ones(len(coo[0]))).to(cls.device)

            X_batch = torch.sparse.FloatTensor(
                i, v, torch.Size([batch_size, nb_steps, nb_units])
            ).to(cls.device)
            y_batch = torch.tensor(labels_[batch_index], device=cls.device)

            yield X_batch.to(device=cls.device), y_batch.to(device=cls.device)

            counter += 1

    @classmethod
    def _get_pytorch_device(cls):
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

        return device

    @classmethod
    def get_download_FashionMNIST(cls):
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

        return x_train, y_train, x_test, y_test

    @classmethod
    def get_layers_weight_list(cls):
        weight_list = []

        # 입력 -> 은닉층 초기값 지정
        w1 = torch.empty(
            (cls.nb_inputs, cls.nb_hidden[0]),
            device=cls.device,
            dtype=cls.dtype,
            requires_grad=True,
        )
        torch.nn.init.normal_(
            w1, mean=0.0, std=cls.weight_scale / np.sqrt(cls.nb_inputs)
        )
        weight_list.append(w1)

        for layer in range(1, len(cls.nb_hidden)):
            w2 = torch.empty(
                (cls.nb_inputs, cls.nb_hidden[layer]),
                device=cls.device,
                dtype=cls.dtype,
                requires_grad=True,
            )
            torch.nn.init.normal_(
                w2, mean=0.0, std=cls.weight_scale / np.sqrt(cls.nb_inputs)
            )
            weight_list.append(w2)

        return weight_list

    @classmethod
    def run_snn(cls, inputs, weight_list):
        # 입력층 -> 첫 번째 은닉층 계산
        h = torch.einsum(
            "abc,cd->abd", (inputs, weight_list[0])
        )  # weight_list는 은닉층 가중치 리스트
        syn = torch.zeros(  # TODO Error 수정
            (cls.batch_size, cls.nb_hidden_list[0]), device=cls.device, dtype=cls.dtype
        )
        mem = torch.zeros(
            (cls.batch_size, cls.nb_hidden_list[0]), device=cls.device, dtype=cls.dtype
        )  # 첫 번째 은닉층

        mem_rec = []
        spk_rec = []

        # 은닉층들의 활동 계산
        for t in range(cls.nb_steps):
            for layer_count in range(len(weight_list)):  # 여러 은닉층을 고려한 반복문
                mthr = mem - 1.0
                out = cls.spike_fn(mthr)
                rst = out.detach()  # We do not want to backprop through the reset

                new_syn = cls.alpha * syn + h[:, t]
                new_mem = (cls.beta * mem + syn) * (1.0 - rst)

                if layer_count == 0:  # 첫 번째 은닉층일 경우 기록
                    mem_rec.append(mem)
                    spk_rec.append(out)

                mem = new_mem
                syn = new_syn

                # 다음 은닉층 입력 계산
                if layer_count < len(weight_list) - 1:  # 마지막 은닉층이 아니면, 다음 은닉층으로 넘김
                    h = torch.einsum(
                        "abc,cd->abd", (out.unsqueeze(1), weight_list[layer_count + 1])
                    )

        mem_rec = torch.stack(mem_rec, dim=1)
        spk_rec = torch.stack(
            spk_rec, dim=1
        )  # spk_rec.shape: (100, 256, 100), (nb_steps, batch_size, hidden_layer_node)

        # 출력층 계산
        h_out = torch.einsum("abc,cd->abd", (spk_rec, weight_list[-1]))
        flt = torch.zeros(
            (cls.batch_size, cls.nb_outputs), device=cls.device, dtype=cls.dtype
        )
        out = torch.zeros(
            (cls.batch_size, cls.nb_outputs), device=cls.device, dtype=cls.dtype
        )
        out_rec = [out]
        for t in range(cls.nb_steps):
            new_flt = cls.alpha * flt + h_out[:, t]
            new_out = cls.beta * out + flt

            flt = new_flt
            out = new_out

            out_rec.append(out)

        out_rec = torch.stack(out_rec, dim=1)
        other_recs = [mem_rec, spk_rec]

        return out_rec, other_recs

    @classmethod
    def train(cls, x_data, y_data, lr=1e-3, nb_epochs=10):
        params = cls.weight_list  # 이제 weight_list에 출력층 가중치도 포함
        optimizer = torch.optim.Adamax(params, lr=lr, betas=(0.9, 0.999))

        log_softmax_fn = torch.nn.LogSoftmax(dim=1)
        loss_fn = torch.nn.NLLLoss()

        loss_hist = []
        for e in range(nb_epochs):
            local_loss = []
            for x_local, y_local in cls.sparse_data_generator(
                x_data, y_data, cls.batch_size, cls.nb_steps, cls.nb_inputs
            ):
                output, recs = cls.run_snn(x_local.to_dense(), cls.weight_list)
                _, spks = recs  # 막전위 기록, 스파이크 기록
                m, _ = torch.max(output, 1)
                log_p_y = log_softmax_fn(m)

                reg_loss = 1e-5 * torch.sum(spks)  # L1 loss on total number of spikes
                reg_loss += 1e-5 * torch.mean(
                    torch.sum(torch.sum(spks, dim=0), dim=0) ** 2
                )  # L2 loss on spikes per neuron

                loss_val = loss_fn(log_p_y, y_local) + reg_loss

                optimizer.zero_grad()
                loss_val.backward()
                optimizer.step()
                local_loss.append(loss_val.item())

            mean_loss = np.mean(local_loss)
            print("Epoch %i: loss=%.5f" % (e + 1, mean_loss))
            loss_hist.append(mean_loss)

        return loss_hist
