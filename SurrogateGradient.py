import torch


class SurrogateGradientSpike(torch.autograd.Function):
    scale = 100.0  # 기울기 정도를 조절

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        out = torch.zeros_like(input)
        out[input > 0] = 1.0  # 스파이크 여부를 판단

        return out

    @staticmethod
    def backward(ctx, grad_output):
        (input,) = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad = grad_input / (SurrogateGradientSpike.scale * torch.abs(input) + 1.0) ** 2

        return grad
