import torch
import torch.nn as nn
from torch.nn import functional as F

# hyperparameters
batch_size = 64 # how many independent sequences will we process in parallel?    # 并行处理的序列数Number of parallel sequences: 这里指的是把训练文本拆分处理训练，所以这是可以同时训练的拆分出来的batch，GPU的并行处理parallel就是这个功能。
block_size = 256 # what is the maximum context length for predictions?    # 上下文窗口大小 Context window size：attention机制下，输出预测下一个token的时候可以参考的其他token数量，简单理解为transformer能“看”到的上下文长度。越大能看到的上下文越多。
max_iters = 5000    # 最大迭代次数Maximum iterations： 训练过程中的迭代次数，每一次迭代更新一次weights/parameters
eval_interval = 500    # 评估间隔 Evaluation interval： 在训练中查看cost的间隔，可以帮助直观看到训练效果。
learning_rate = 3e-4    # 学习率Learning rate：在backprop训练的时候，得到的gradient descent乘以学习率用来更新权重weights，公式：old weight = new weight - learning rate * gradient descent。3e-4是transformer的常用值，太大会震荡，太小收敛满。
device = 'cuda' if torch.cuda.is_available() else 'cpu' # 选择Cuda还是CPU，根据具体训练设备决定
eval_iters = 200 # 评估迭代次数 Evaluation iterations：这个是训练时的参数，每200个batch拿出来算一下loss，减少评估结果时候的随机波动。
n_embd = 384 # 嵌入维度 Embedding dimension： 这个值代表token向量的大小，384的意思是每个token有384维的连续向量，维度越高包含越多语义信息。这个值可以调整，根据设备的配置决定。越小计算速度越快，越大越能训练不同的语义结构
n_head = 6   # 注意力头数 Attention heads：perallet训练需要多头，GPT3 有96个头，不同的头可以训练出不同的pattern，最后把结果拼接起来。6头意味着学习6种不同的注意力模式，比如不同的位置的语法，语义，环境之类的。可以简单理解为头数越多，训练得出的模型表现越好。在简化版的训练当中，小模型使用小量的头就可以
n_layer = 6    # 网络层数 Number of layers： transformer的堆叠次数，训练网络的层级，一般越大的模型，MLP的层数越多，GPT3 每个头有96层
dropout = 0.2    # 丢弃率Dropout rate ：防止overfitting，训练时随机丢弃20%的neuron，在测试评估的时候使用全部的neuron
# ------------

torch.manual_seed(1337) # 随机种子 | Random seed： 计算机不能生成真的随机数，随便设置，这里的1337是惯例。

# 这里是Data Pipeline，读取文本
# wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

#encoding/decoding的部分
# here are all the unique characters that occur in this text
chars = sorted(list(set(text)))      # 字符去重排序
vocab_size = len(chars)    # 词表大小
# create a mapping from characters to integers 
stoi = { ch:i for i,ch in enumerate(chars) }    # 字符-》索引
itos = { i:ch for i,ch in enumerate(chars) }    # 索引 -》字符
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers         # encoder 函数 （需整理）
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string        # decoder函数  （需整理）

#下面是数据分割，分割成90%训练，10%测试
# Train and test splits
data = torch.tensor(encode(text), dtype=torch.long)      # 把text变成tensor
n = int(0.9*len(data)) # first 90% will be train, rest val    # 分割数据
train_data = data[:n]    # 训练数据
val_data = data[n:]    # 测试数据

# data loading
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

#这里是模型的部分
class Head(nn.Module):
    """ one head of self-attention """ ##self-attention从这里开始，这里是attention head 编写

    def __init__(self, head_size):
        super().__init__()            # K Q V 投影，不考虑bias
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))    # 这里是做positional masking

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # input of size (batch, time-step, channels)
        # output of size (batch, time-step, head size)
        B,T,C = x.shape    # attention 计算
        k = self.key(x)   # (B,T,hs)    
        q = self.query(x) # (B,T,hs)
        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5 # (B, T, hs) @ (B, hs, T) -> (B, T, T)    # 这里是scaled dot product
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)    #    masking （需整理）
        wei = F.softmax(wei, dim=-1) # (B, T, T)    #使用softmask更新调整权重
        wei = self.dropout(wei)  
        # perform the weighted aggregation of the values
        v = self.value(x) # (B,T,hs)    #给V值做变换
        out = wei @ v # (B, T, T) @ (B, T, hs) -> (B, T, hs)   
        return out

class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """    #这边写的是多头

    def __init__(self, num_heads, head_size):   
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)    #投影层
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # 多头输出拼接
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))    # 丢掉一些投影结果 dropout变量防止过拟合
        return out

class FeedFoward(nn.Module):
    """ a simple linear layer followed by a non-linearity """    #参考attention is all you need里面的feedforward 层

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            #这里需要再整理，这里使用了一个经典算法，linear，再ReLU，再linear
            nn.Linear(n_embd, 4 * n_embd),    #扩展
            nn.ReLU(),                # activation
            nn.Linear(4 * n_embd, n_embd),    #压缩
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """ Transformer block: communication followed by computation """ # 这边是transformer构建

    def __init__(self, n_embd, n_head):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)        #多头自注意力
        self.ffwd = FeedFoward(n_embd)                #前馈网络
        self.ln1 = nn.LayerNorm(n_embd)    
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class GPTLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd) # final layer norm
        self.lm_head = nn.Linear(n_embd, vocab_size)

        # better init, not covered in the original GPT video, but important, will cover in followup video
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx) # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T,C)
        x = tok_emb + pos_emb # (B,T,C)
        x = self.blocks(x) # (B,T,C)
        x = self.ln_f(x) # (B,T,C)
        logits = self.lm_head(x) # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -block_size:]
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx

model = GPTLanguageModel()
m = model.to(device)
# print the number of parameters in the model
print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):

    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # sample a batch of data
    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))
#open('more.txt', 'w').write(decode(m.generate(context, max_new_tokens=10000)[0].tolist()))
