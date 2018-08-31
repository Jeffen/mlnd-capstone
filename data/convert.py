"""
A sample of convert the cifar100 dataset to 224 * 224 size train\val data.
"""
import cv2
import os
from keras.datasets import cifar100


def convert():
    train = 'data/train//'
    valid = 'data/validation//'
    test = 'data/test//'

    (X_train, y_train), (X_test, y_test) = cifar100.load_data(label_mode='fine')

    for i, val in enumerate(X_train[5000:]):
        x = val
        y = y_train[i+5000]
        path = train + str(y[0])
        x = cv2.resize(x, (224, 224), interpolation=cv2.INTER_CUBIC)
        if not os.path.exists(path):
            os.makedirs(path)
        cv2.imwrite(path + '//' + str(i) + '.jpg', x)

    for i, val in enumerate(X_train[:5000]):
        x = val
        y = y_train[i]
        path = valid + str(y[0])
        x = cv2.resize(x, (224, 224), interpolation=cv2.INTER_CUBIC)
        if not os.path.exists(path):
            os.makedirs(path)
        cv2.imwrite(path + '//' + str(i) + '.jpg', x)

    for i, val in enumerate(X_test):
        x = X_test[i]
        y = y_test[i]
        path = test + str(y[0])
        x = cv2.resize(x, (224, 224), interpolation=cv2.INTER_CUBIC)
        if not os.path.exists(path):
            os.makedirs(path)
        cv2.imwrite(path + '//' + str(i) + '.jpg', x)


if __name__ == '__main__':
    convert()