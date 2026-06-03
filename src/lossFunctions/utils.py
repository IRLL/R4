import torch


def cost_fn(x, y, power):
    delta = x[:, None] - y[None, :]
    if power == 1.0:
        cost = torch.abs(delta)
    elif power == 2.0:
        cost = delta ** 2.0
    else:
        cost = torch.abs(delta) ** power
    return cost


def get_distr(u, v, K):
    return u[:, None] * K * v[None, :]


def Sinkhorn(a, b, x, y, eps, power, nu):
    C = cost_fn(x, y, power)
    K = torch.exp(-C / eps)

    u = torch.ones_like(x)

    v = b / (K.T @ u)
    u = a / (K @ v)

    while torch.abs(v * (K.T @ u) - b).sum() > nu:
        v = b / (K.T @ u)
        u = a / (K @ v)

    return u, v, K


def Rank_Sort(a, b, x, y, eps=0.1, power=2, nu=1e-5):
    u, v, K = Sinkhorn(a, b, x, y, eps, power, nu)

    b_hat = torch.cumsum(b, dim=0)
    n = x.shape[0]

    R_tilda = (n * (a ** -1)) * u * (K @ (v * b_hat))
    S_tilda = (b ** -1) * v * (K.T @ (u * x))

    return R_tilda, S_tilda


def get_delta(cost, alpha, betta, b, eps):
    b_bar = torch.exp(-(cost.T - alpha.T - betta) / eps).sum(dim=1)
    return torch.abs(b - b_bar).sum()


def soft_min(M, eps):
    return -eps * torch.logsumexp(-M / eps, dim=1, keepdim=True)


def log_Sinkhorn(a, b, x, y, eps=1e-2, power=2, nu=1e-5):
    C = cost_fn(x, y, power)

    alpha = torch.zeros((x.shape[0], 1), device=x.device, dtype=x.dtype)
    betta = torch.zeros((y.shape[0], 1), device=y.device, dtype=y.dtype)

    while get_delta(C, alpha, betta, b, eps) > nu:
        alpha = eps * torch.log(a[:, None]) + soft_min(C - alpha - betta.T, eps) + alpha
        betta = eps * torch.log(b[:, None]) + soft_min(C.T - alpha.T - betta, eps) + betta

    return alpha, betta, C


def Id(x):
    return x


def squash(x, scale=1.0, min_std=1e-10):
    mu = torch.mean(x)
    std = torch.std(x)
    s = scale * torch.sqrt(torch.tensor(3.0, device=x.device)) / torch.pi * torch.maximum(
        std, torch.tensor(min_std, device=x.device)
    )
    return torch.sigmoid((x - mu) / s)


def Rank_Sort_log(a, b, x, y, eps=1e-2, power=2, nu=1e-5, g=Id):
    alpha, betta, C = log_Sinkhorn(a, b, g(x), y, eps, power, nu)

    b_hat = torch.cumsum(b, dim=0)

    transport = torch.exp(-(C - alpha - betta.T) / eps)

    R_tilda = len(x) * (a ** -1) * (transport @ b_hat)
    S_tilda = (b ** -1) * (transport.T @ x[:, None])

    return R_tilda.flatten(), S_tilda.flatten()
