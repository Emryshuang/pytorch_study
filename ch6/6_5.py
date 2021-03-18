import torch
from torch import nn
import d2l_pytorch.d2l as d2l

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

(corpus_indices, char_to_idx, idx_to_char, vocab_size) = d2l.load_data_jay_lyrics()
# 1 定义模型

num_hiddens = 256
run_layer = nn.RNN(input_size=vocab_size, hidden_size=num_hiddens)

num_steps = 35
batch_size = 2

state = None
X = torch.rand(num_steps, batch_size, vocab_size)
Y, state_new = run_layer(X, state)
print(Y.shape, len(state_new), state_new[0].shape)
# 2 训练模型

model = d2l.RNNModel(run_layer, vocab_size).to(device)
print(d2l.predict_rnn_pytorch('分开', 10, model, vocab_size, device, idx_to_char, char_to_idx))

num_epochs, batch_size, lr, clipping_theta = 250, 32, 1e-3, 1e-2
pred_period, pred_len, prefixes = 50, 50, ['分开', '不分开']

d2l.train_and_predict_rnn_pytorch(model, num_hiddens, vocab_size, device, corpus_indices, idx_to_char, char_to_idx, num_epochs,
                        num_steps, lr, clipping_theta, batch_size, pred_period, pred_len, prefixes)
