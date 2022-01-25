import json
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pickle


plt.figure(figsize=(10, 6))
alg_to_batch = {'Vaidya': [128, 256], 'SGD': [64, 128, 4]}
for alg, batches in alg_to_batch.items():
    for batch in batches:
        fname = f'{alg}_batch_{batch}.pickle'
        with open(fname, 'rb') as f:
            losses = pickle.load(f)
        label = f'{alg}, batch size {batch}'
        if alg == 'SGD':
            label += ' , step size ' + ('0.01' if batch == 4 else '0.1')
        if alg == 'SGD' and batch == 128:
            plt.plot(losses, label=label, linestyle='--')
        elif alg == 'SGD' and batch == 64:
            plt.plot(losses, label=label, linewidth=3, alpha=0.8)
        else:
            plt.plot(losses, label=label)


# plt.title('BCE Loss for SGD and Vaidya method')
plt.xlabel('Steps')
plt.ylabel('Loss')
plt.legend()
plt.ylim(0.48, 0.72)
plt.xlim(0, 3633)
# plt.plot(range(1, len(vaidya_loss) + 1), vaidya_loss, 'r')
# plt.plot(range(1, len(sgd_loss) + 1), sgd_loss, 'b')
plt.gca().xaxis.set_major_locator(mticker.MultipleLocator(300))
plt.grid(True, linestyle='-', color='0.75')
plt.savefig('vaidya_and_sgd_loss2.png', bbox_inches='tight')