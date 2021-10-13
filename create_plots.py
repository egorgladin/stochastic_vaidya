import json
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker


with open("vaidya_loss.json", 'r') as f:
    vaidya_loss = json.load(f)

with open("sgd_loss.json", 'r') as f:
    sgd_loss = json.load(f)


plt.figure()

plt.title('BCE Loss for SGD and Vaidya method')
plt.xlabel('Steps')
plt.ylabel('Loss')
plt.plot(range(1, len(vaidya_loss) + 1), vaidya_loss, 'r')
plt.plot(range(1, len(sgd_loss) + 1), sgd_loss, 'b')
plt.gca().xaxis.set_major_locator(mticker.MultipleLocator(1))
plt.grid(True, linestyle='-', color='0.75')
plt.savefig('vaidya_and_sgd_loss.png')