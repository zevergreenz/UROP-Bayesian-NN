import numpy as np
import matplotlib.pyplot as plt

training_size, nn_random = np.load('nn_random.npy')
training_size, nn_max_variance = np.load('nn_max_variance.npy')

fig, ax = plt.subplots(1, 1, figsize=(12, 9))

ax.get_xaxis().tick_bottom()
ax.get_yaxis().tick_left()

# plt.xticks(range(300, 1201, 100), fontsize=14)
plt.xticks(range(0, 41, 2), fontsize=11)
ax.set_xlabel('Training Size', fontsize=12)
ax.set_ylabel('Mean Square Error', fontsize=12)

plt.plot(training_size, nn_random, marker='v', lw=1., color='#ff7f0e', label='random')
plt.plot(training_size, nn_max_variance, marker='v', lw=1., color='#aec7e8', label='maximum predictive variance')

plt.legend(bbox_to_anchor=(0.05, 0.1), loc=2, borderaxespad=0.)

plt.savefig('boston_housing_mse.png', dpi=300)