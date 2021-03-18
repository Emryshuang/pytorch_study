import time
import math
import numpy as np
import torch
from torch import nn, optim
import torch.nn.functional as F
import d2l_pytorch.d2l as d2l

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

(corpus_indices, char_to_idx, idx_to_char, vocab_size) = d2l.load_data_jay_lyrics()


# 1 ONE-HOT向量

def one_hot(x, n_class, dtype=torch.float32):
    x = x.long()
    res = torch.zeros(x.shape[0], n_class, dtype=dtype)
    res.scatter_(1, x.view(-1, 1), 1)
    return res


x = torch.tensor([0, 2])
print(one_hot(x, vocab_size))

X = torch.arange(10).view(2, 5)
inputs = d2l.to_onehot(X, vocab_size)
#print(len(inputs), inputs[0].shape)
#print(inputs)

# 2 初始化模型参
num_inputs, num_hiddens, num_outputs = vocab_size, 256, vocab_size
print('will use', device)


def get_params():
    def _one(shape):
        ts = torch.tensor(np.random.normal(0, 0.01, size=shape), device=device, dtype=torch.float32)
        return torch.nn.Parameter(ts, requires_grad=True)

    W_xh = _one((num_inputs, num_hiddens))
    W_hh = _one((num_hiddens, num_hiddens))
    b_h = torch.nn.Parameter(torch.zeros(num_hiddens, device=device, requires_grad=True))
    W_hq = _one((num_hiddens, num_outputs))
    b_q = torch.nn.Parameter(torch.zeros(num_outputs, device=device, requires_grad=True))

    return nn.ParameterList([W_xh, W_hh, b_h, W_hq, b_q])


# 3 定义模型
def init_run_state(bath_size, num_hiddens, device):
    return (torch.zeros((bath_size, num_hiddens), device=device),)


def run(inputs, state, params):
    W_xh, W_hh, b_h, W_hq, b_q = params
    H, = state
    outputs = []
    for X in inputs:
        H = torch.tanh(torch.matmul(X, W_xh) + torch.matmul(H, W_hh) + b_h)
        Y = torch.matmul(H, W_hq) + b_q
        outputs.append(Y)
    return outputs, (H,)


state = init_run_state(X.shape[0], num_hiddens, device)
inputs = d2l.to_onehot(X.to(device), vocab_size)
params = get_params()
outputs, state_new = run(inputs, state, params)
#print(len(outputs), outputs[0].shape, state_new[0].shape)

# 4 定义预测函数
#print(d2l.predict_run('分开', 10, run, params, init_run_state, num_hiddens, vocab_size, device, idx_to_char, char_to_idx))

# 5 裁剪梯度

# 6 困惑度

# 7 定义模型训练函数

# 8 训练模型并创作歌词
# 8 训练模型并创作歌词

num_epochs, num_steps, batch_size, lr, clipping_theta = 250, 35, 32, 1e2, 1e-2
pred_period, pred_len, prefixes = 50, 50, ['分开', '不分开']


d2l.train_and_predict_rnn(run, get_params, init_run_state, num_hiddens, vocab_size, device, corpus_indices, idx_to_char,
                          char_to_idx, True, num_epochs, num_steps, lr, clipping_theta, batch_size, pred_period,
                          pred_len, prefixes)

d2l.train_and_predict_rnn(run, get_params, init_run_state, num_hiddens, vocab_size, device, corpus_indices, idx_to_char,
                          char_to_idx, False, num_epochs, num_steps, lr, clipping_theta, batch_size, pred_period,
                          pred_len, prefixes)
