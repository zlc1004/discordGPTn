import model
import torch
import tqdm
from rich import print
from rich.panel import Panel
from rich.padding import Padding
import os
import pickle
# hyperparameters
batch_size = 64  # how many independent sequences will we process in parallel
block_size = 256  # the maximum context length for predictions
max_iters = 1000000  # how many iteration during training
eval_interval = 750  # how often to evaluate training
learning_rate = 3e-4
# device = "cpu"
device = model.get_device()
eval_iters = 100
n_embd = 384
n_head = 6
n_layer = 6
dropout = 0.2
# data loading


def get_batch(data, block_size, batch_size, device):
    # generate a small batch of data of inputs x and targets y
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y


@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for label, data in {"train": train_data, "val": val_data}.items():
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(data, block_size, batch_size, device)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[label] = losses.mean()
    model.train()
    return out


# read it in to inspect it
with open('data/input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)

tokenizer = model.Tokenizer(chars)

data = torch.tensor(tokenizer.encode(text), dtype=torch.long)

n = int(0.9*len(data))  # first 90% will be train, rest is validation
train_data = data[:n]
val_data = data[n:]

model = model.GPTLanguageModel(vocab_size=vocab_size, n_embd=n_embd,
                               n_layer=n_layer, n_head=n_head, block_size=block_size, dropout=dropout)
model.to(device)  # use GPU device if available
# print the number of parameters in the model

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

print(Panel.fit(Padding("[yellow][bold]"+(str(sum(p.numel() for p in model.parameters())/1e6) +
      '[/yellow][/bold] M parameters')+"\n[bold]to: "+device+"[/bold]", (2, 5), style="on blue", expand=False)))


for iter in tqdm.tqdm(range(max_iters)):
    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss()
        print(
            f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # sample a batch of data
    xb, yb = get_batch(train_data, block_size, batch_size, device)

    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

with open("model/model.pkl", "wb") as f:
    pickle.dump(model, f)

# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
generated_text = tokenizer.decode(model.generateText(
    context, max_new_tokens=2000)[0].tolist())
print(generated_text)

if not os.path.exists("data"):
    os.mkdir("data")
with open('data/output.txt', 'w', encoding="utf-8") as f:
    f.write(generated_text)

if not os.path.exists("model"):
    os.mkdir("model")

with open("model/tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)
