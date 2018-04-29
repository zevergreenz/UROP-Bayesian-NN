import numpy as np
import matplotlib.pyplot as plt

training_size, cnn_random = np.load('cnn_random.npy')
cnn_max_entropy = np.load('cnn_max_entropy.npy')
cnn_max_meanvar = np.load('cnn_max_meanvar.npy')
cnn_max_bald = np.load('cnn_max_bald.npy')

fig, ax = plt.subplots(1, 1, figsize=(12, 9))

ax.get_xaxis().tick_bottom()
ax.get_yaxis().tick_left()

# plt.xticks(range(300, 1201, 100), fontsize=14)
plt.yticks(range(0, 101, 5), fontsize=11)
ax.set_xlabel('Training Size', fontsize=12)
ax.set_ylabel('Accuracy', fontsize=12)

plt.plot(training_size, cnn_random, marker='v', lw=1., color='#ff7f0e', label='random')
plt.plot(training_size, cnn_max_entropy, marker='v', lw=1., color='#1f77b4', label='maximum entropy')
plt.plot(training_size, cnn_max_meanvar, marker='v', lw=1., color='#ff0000', label='maximum mean variance')
plt.plot(training_size, cnn_max_bald, marker='v', lw=1., color='#008000', label='maximum BALD')

plt.legend(bbox_to_anchor=(0.7, 0.23), loc=2, borderaxespad=0.)

# plt.show()
plt.savefig('mnist_cnn_accuracy.png', dpi=300)