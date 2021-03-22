import zipfile
import d2l_pytorch.d2l as d2l

# 1 读取数据集
with zipfile.ZipFile('../data/ch6/jaychou_lyrics.txt.zip') as zin:
    with zin.open('jaychou_lyrics.txt') as f:
        corpus_chars = f.read().decode('utf-8')

print(corpus_chars[:40])
corpus_chars = corpus_chars.replace('\n', ' ').replace('\r', ' ')
# print(corpus_chars[:10000])
corpus_chars = corpus_chars[:10000]

# 2 建⽴字符索引
idx_to_char = list(set(corpus_chars))
char_to_idx = dict([(char, i) for i, char in enumerate(idx_to_char)])
vocab_size = len(char_to_idx)
print(vocab_size)

corpus_chars = [char_to_idx[char] for char in corpus_chars]
sample = corpus_chars[:20]
print('chars', ''.join([idx_to_char[idx] for idx in sample]))
print('indices', sample)

# 3 时序数据的采样

my_seq = list(range(30))
for X, Y in d2l.data_iter_random(my_seq, batch_size=2, num_steps=6):
    print('x:', X, '\nY', Y, '\n')

for X, Y in d2l.data_iter_consecutive(my_seq, batch_size=2, num_steps=6):
    print('x:', X, '\nY', Y, '\n')
