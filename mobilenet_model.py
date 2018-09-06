"""
Train the MobileNet V2 model
"""
import os
import sys
import argparse
import pandas as pd
import numpy as np

from mobilenet_model import MobileNetv2

from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Conv2D, Reshape, Activation
from keras.models import Model

from sklearn.metrics import confusion_matrix

def fine_tune(model):
    """Re-build model with current num_classes.

    # Arguments
        model, Model, The model structure.
    """
    model.load_weights('saved_models/weights.best.mbnetv2.hdf5')

    x = model.get_layer('Dropout').output
    x = Conv2D(100, (1, 1), padding='same')(x)
    x = Activation('softmax', name='softmax')(x)
    output = Reshape((100,))(x)

    model = Model(inputs=model.input, outputs=output)

    return model


def train(batch, epochs, num_classes, size, weights, train_tensors, train_targets, valid_tensors, valid_targets):
    """Train the model.

    # Arguments
        batch: Integer, The number of train samples per batch.
        epochs: Integer, The number of train iterations.
        num_classes, Integer, The number of classes of dataset.
        size: Integer, image size.
        weights, Boolean, The pre_trained model weights.
    """
    if not os.path.exists('saved_models'):
        os.makedirs('saved_models')

    if weights:
        model = MobileNetv2((size, size, 3), num_classes)
        model = fine_tune(model)
    else:
        model = MobileNetv2((size, size, 3), num_classes)

    opt = Adam()
    earlystop = EarlyStopping(monitor='val_acc', patience=15, verbose=0, mode='auto')
    checkpointer = ModelCheckpoint(filepath='saved_models/weights.best.mbnetv2.hdf5', 
                                verbose=1, save_best_only=True)

    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

    hist = model.fit(train_tensors, train_targets,
            validation_data=(valid_tensors, valid_targets),
            epochs=epochs, verbose=2, batch_size=batch, callbacks=[earlystop, checkpointer])

    df = pd.DataFrame.from_dict(hist.history)
    df.to_csv('saved_models/hist_mbnet.csv', encoding='utf-8', index=False)
    
    
def score(size, num_classes, test_tensors, test_targets):
    model = MobileNetv2((size, size, 3), num_classes)
    model.load_weights('saved_models/weights.best.mbnetv2.hdf5')
    # get index of predicted dog breed for each image in test set
    predictions = [np.argmax(model.predict(np.expand_dims(tensor, axis=0))) for tensor in test_tensors]

    confusion_matrix(np.argmax(test_targets, axis=1), np.array(predictions), labels=range(99))
    # report test accuracy
    test_accuracy = 100*np.sum(np.array(predictions)==np.argmax(test_targets, axis=1))/len(predictions)
    print('Test accuracy: %.4f%%' % test_accuracy)
