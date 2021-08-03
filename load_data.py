from scipy.io import loadmat
import matplotlib.pyplot as plt
from random import randrange

# !wget http://ufldl.stanford.edu/housenumbers/train_32x32.mat
# !wget http://ufldl.stanford.edu/housenumbers/test_32x32.mat
train = loadmat('svhn/train_32x32.mat')
test = loadmat('test_32x32.mat')

for label in train['y']:
    if label[0] == 10:
        label[0] = 0

for label in test['y']:
    if label[0] == 10:
        label[0] = 0


for i in range(5):
    random_index = randrange(train['X'].shape[3])
    training_sample = train['X'][:, :, :, random_index]
    training_label = train['y'][random_index]

    plt.imshow(training_sample)
    plt.show()
    print(f"Digit label: {training_label}")



