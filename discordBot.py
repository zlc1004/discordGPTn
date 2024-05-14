import discord
import model
import torch
import pickle
from rich import print
from rich.panel import Panel
from rich.padding import Padding
from rich.live import Live

max_new_tokens=100

intents = discord.Intents.default()
intents.message_content = True

device = model.get_device()
Panel.fit(Padding("to device", (2, 5), style="on blue", expand=False))
print(Panel.fit(Padding("to [bold]"+device+"[/bold]", (2, 5), style="on blue", expand=False)))

with open("model/model.pkl", "rb") as f:
    gpt_model = pickle.load(f,fix_imports=False)
    gpt_model.to(device)
with open("model/tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)
print(Panel("model loaded"))

client = discord.Client(intents=intents)
chanels=[1239710224332488704,1239745226298622052,1239741201293250631]

@client.event
async def on_ready():
    global live
    print("We have logged in as" ,client.user)

messageHistory=""

@client.event
async def on_message(message: discord.Message):
    with Live(refresh_per_second=10) as live:
        global messageHistory
        if message.author == client.user:
            return
        if message.channel.id in chanels:
            if str(message.content)=="!save":
                with open("chat.log", "w",encoding="utf-8") as f:
                    f.write(messageHistory)
                await message.reply("Saved")
                return
            if str(message.content)=="!clear":
                messageHistory=""
                await message.reply("Cleared")
                return
            messageHistory+= "\n"+str(message.content)+"\n"
            encodedMessage = tokenizer.encode(messageHistory)
            context = torch.zeros((1, len(encodedMessage)), dtype=torch.long, device=device)
            for i in range(len(encodedMessage)):
                context[0, i] = (encodedMessage)[i]
            # print(context)
            generated_textObj=gpt_model.generate(context, max_new_tokens=max_new_tokens)
            for i in range(max_new_tokens):
                liveText=message.content+'\n'+tokenizer.decode(next(generated_textObj)[0].tolist())
                if str(liveText)[-1]=="\n":
                    break
            out=tokenizer.decode(next(generated_textObj)[0].tolist())
            messageHistory=out
            messageHistory=messageHistory.split("\n")
            tmp=[]
            for i in messageHistory:
                if len(i)>1:
                    tmp.append(i)
            messageHistory="\n".join(tmp)
            live.update(Panel.fit(messageHistory))
            await message.reply(messageHistory.split("\n")[-1])

client.run('TokenRemoved')
