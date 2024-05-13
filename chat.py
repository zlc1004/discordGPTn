import model
import torch
import pickle

device = "cpu" # model.get_device()
print("to", device)

with open("model/model.pkl", "rb") as f:
    gpt_model = pickle.load(f)
    gpt_model.to(device)
with open("model/tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)
    
context = torch.zeros((1, 1), dtype=torch.long, device=device)
generated_text = tokenizer.decode(gpt_model.generate(context, max_new_tokens=2000)[0].tolist())
print(generated_text)

with open('output.txt', 'w', encoding="utf-8") as f:
    f.write(generated_text)
