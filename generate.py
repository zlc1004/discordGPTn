from rich.live import Live
from rich.padding import Padding
from rich.panel import Panel
from rich import print
import pickle
import torch
import model
max_new_tokens = 1000


device = model.get_device()
Panel.fit(Padding("to device", (2, 5), style="on blue", expand=False))
print(Panel.fit(Padding("to [bold]"+device +
      "[/bold]", (2, 5), style="on blue", expand=False)))

with open("model/model1.pkl", "rb") as f:
    gpt_model = pickle.load(f, fix_imports=False)
    gpt_model.to(device)
with open("model/tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)
panel = Panel("model loaded")

message = "yo"

encodedMessage = tokenizer.encode(message)
context = torch.zeros(
    (1, len(message)), dtype=torch.long, device=device)
for i in range(len(encodedMessage)):
    context[0, i] = (encodedMessage)[i]
# print(context)
generated_textObj = gpt_model.generate(
    context, max_new_tokens=max_new_tokens)
with Live(refresh_per_second=10, vertical_overflow="visible",screen=False,transient=True) as live:
    for i in range(max_new_tokens):
        liveText = tokenizer.decode(
            next(generated_textObj)[0].tolist())
        liveText = Panel.fit(liveText)
        live.update(liveText)
message = tokenizer.decode(next(generated_textObj)[0].tolist())
message = message.split("\n")
tmp = []
for i in message:
    if len(i) > 1:
        tmp.append(i)
message = "\n".join(tmp)


with open("chat.log", "w", encoding="utf-8") as f:
    f.write(message)

if input("upload to train data?y/N") == "y":
    with open("data/input.txt", "a", encoding="utf-8") as f:
        f.write(message)
