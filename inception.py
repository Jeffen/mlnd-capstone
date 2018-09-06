"""
Train the Inception model
"""
import os
import pandas as pd
import numpy as np

from keras.applications.inception_v3 import InceptionV3

from keras.models import Model
from keras import backend as K
from keras.layers import Dense, GlobalAveragePooling2D, Dropout
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping
from keras.optimizers import SGD


def Inception(input_shape, k):
    """InceptionV3
    This function defines a InceptionV3 architectures.

    # Arguments
        input_shape: An integer or tuple/list of 3 integers, shape
            of input tensor.
        k: Integer, number of classes.
    # Returns
        InceptionV3 model.
    """
    # create the base pre-trained model
    base_model = InceptionV3(input_shape=input_shape,
                             weights='imagenet', include_top=False)

    # add a global spatial average pooling layer
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    # let's add a fully-connected layer
    x = Dense(1024, activation='relu')(x)
    # add a dropout layer
    x = Dropout(0.3)(x)
    # and a logistic layer -- let's say we have 100 classes
    predictions = Dense(k, activation='softmax')(x)

    # this is the model we will train
    model = Model(inputs=base_model.input, outputs=predictions)

    # first: train only the top layers (which were randomly initialized)
    # i.e. freeze all convolutional InceptionV3 layers
    for layer in base_model.layers:
        layer.trainable = False

    return model


def generate(batch, train_tensors, train_targets, valid_tensors, valid_targets):
    """Data generation and augmentation

    # Arguments
        batch: Integer, batch size.
        size: Integer, image size.

    # Returns
        train_generator: train set generator
        validation_generator: validation set generator
        count1: Integer, number of train set.
        count2: Integer, number of test set.
    """

    #  Using the data Augmentation in traning data
    datagen1 = ImageDataGenerator(
        shear_range=0.2,
        rotation_range=90,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True)

    datagen2 = ImageDataGenerator(rotation_range=90)
    
    train_generator = datagen1.flow(
        train_tensors,
        train_targets,
        batch_size=batch)

    validation_generator = datagen2.flow(
        valid_tensors,
        valid_targets,
        batch_size=batch)
    
    count1 = len(train_tensors)
    count2 = len(valid_tensors)
    return train_generator, validation_generator, count1, count2


def fine_tune(model):
    """Re-build model with current num_classes.

    # Arguments
        num_classes, Integer, The number of classes of dataset.
        tune, String, The pre_trained model weights.
        model, Model, The model structure.
    """
    model.load_weights('saved_models/weights.best.inception.hdf5')

    # we chose to train the top 2 inception blocks, i.e. we will freeze
    # the first 249 layers and unfreeze the rest:
    for layer in model.layers[:249]:
        layer.trainable = False
    for layer in model.layers[249:]:
        layer.trainable = True

    return model


def train(batch, epochs, num_classes, size, weights, train_tensors, train_targets, valid_tensors, valid_targets):
    """Train the model.

    # Arguments
        batch: Integer, The number of train samples per batch.
        epochs: Integer, The number of train iterations.
        num_classes, Integer, The number of classes of dataset.
        size: Integer, image size.
        weights, Boolean, Load the pre_trained model weights.
    """
    if not os.path.exists('saved_models'):
        os.makedirs('saved_models')

    if weights:
        model = Inception((size, size, 3), num_classes)
        model = fine_tune(model)
        model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy')
    else:
        model = Inception((size, size, 3), num_classes)
        model.compile(optimizer='rmsprop', loss='categorical_crossentropy')

    
    earlystop = EarlyStopping(monitor='val_loss', patience=15, verbose=1, mode='auto')
    checkpointer = ModelCheckpoint(filepath='saved_models/weights.best.inception.hdf5', 
                                verbose=1, save_best_only=True)

    hist = model.fit(train_tensors, train_targets,
            validation_data=(valid_tensors, valid_targets),
            epochs=epochs, verbose=2, batch_size=batch, callbacks=[earlystop, checkpointer])

    df = pd.DataFrame.from_dict(hist.history)
    df.to_csv('saved_models/incept_hist.csv', encoding='utf-8', index=False)

    """
    # Due to limited computation resources, data augmentation is disabled in this case.
    train_generator, validation_generator, count1, count2 = generate(batch, train_tensors, train_targets, valid_tensors, valid_targets)
    model.fit_generator(train_generator,
                        validation_data=validation_generator,
                        steps_per_epoch=count1 // batch,
                        validation_steps=count2 // batch,
                        epochs=epochs, verbose=2, callbacks=[earlystop, checkpointer])
    """

def score(size, num_classes, test_tensors, test_targets):
    model = Inception((size, size, 3), num_classes)
    model.load_weights('saved_models/weights.best.inception.hdf5')
    # get index of predicted dog breed for each image in test set
    predictions = [np.argmax(model.predict(np.expand_dims(tensor, axis=0))) for tensor in test_tensors]

    # report test accuracy
    test_accuracy = 100*np.sum(np.array(predictions)==np.argmax(test_targets, axis=1))/len(predictions)
    print('Test accuracy: %.4f%%' % test_accuracy)