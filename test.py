from rich import print
from rich.padding import Padding
from rich.panel import Panel
test = Panel.fit(Padding("to device", (2, 5), style="on blue", expand=False))
print(test)
