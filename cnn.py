import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import cv2
import os
import glob
from collections import Counter
import itertools
import pickle

from scipy.misc import imresize
from keras.utils import np_utils
from random import shuffle
from sklearn.utils import resample
from sklearn.utils import class_weight
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras import optimizers
from keras import callbacks
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.xception import preprocess_input

from eda import read_images, targets, train_val_holdout

import pdb



class CNNModel(object):

    def __init__(self,train_folder, validation_folder, holdout_folder, target_size, augmentation_strength=0.2,
                preprocessing=None, batch_size = 8, nb_classes = 4, nb_epoch = 50):
        self.model = Sequential()
        self.train_folder = train_folder
        self.validation_folder = validation_folder
        self.holdout_folder = holdout_folder
        self.target_size = target_size
        self.input_size = self.target_size + (3,)
        self.augmentation_strength = augmentation_strength
        self.batch_size = batch_size
        self.nb_classes = nb_classes
        self.nb_epoch = nb_epoch
        self.train_generator = None
        self.validation_generator = None
        self.holdout_generator = None
        self.preprocessing = preprocessing
        self.history = None

        self.nHoldout = sum(len(files) for _, _, files in os.walk(self.holdout_folder)) #: number of holdout samples
        self.nTrain = sum(len(files) for _, _, files in os.walk(self.train_folder)) #: number of training samples
        self.nVal = sum(len(files) for _, _, files in os.walk(self.validation_folder)) #: number of validation samples

    def conv(self):
        """
        Create a layered Conv/Pooling block
        """

        # first convolutional layer and subsequent pooling
        self.model.add(Convolution2D(32, 3, 3, border_mode='valid', input_shape=(self.input_size), activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))

        # second convolutional layer and subsequent pooling
        self.model.add(Convolution2D(64, 3, 3, border_mode='valid', activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))

        # third convolutional layer
        self.model.add(Convolution2D(128, 3, 3, border_mode='valid', activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))

        # fourth convolutional layer and subsequent pooling
        self.model.add(Convolution2D(128, 3, 3, border_mode='valid', activation='relu', init='glorot_normal'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))

        # flattens images to go into dense layers
        self.model.add(Flatten())
        pdb.set_trace()

        # first dense layer
        self.model.add(Dense(2048, init = 'glorot_normal'))
        self.model.add(Activation('relu'))
        self.model.add(Dropout(0.5))
        pdb.set_trace()

        # second dense layer
        self.model.add(Dense(2048, init= 'glorot_normal'))
        self.model.add(Activation('relu'))
        self.model.add(Dropout(0.5))

        # third dense layer
        self.model.add(Dense(1024, init = 'glorot_normal'))
        self.model.add(Activation('relu'))
        self.model.add(Dropout(0.5))

        # fourth dense layer
        self.model.add(Dense(1024, init = 'uniform'))
        self.model.add(Activation('relu'))
        self.model.add(Dropout(0.5))

        # output layer
        self.model.add(Dense(self.nb_classes, init='glorot_normal'))
        self.model.add(Activation('softmax'))


    def make_generator(self):
        # starting with no augmentation
        train_datagen = ImageDataGenerator(
                        preprocessing_function=self.preprocessing,
                        # rotation_range=50*self.augmentation_strength,
                        # width_shift_range=self.augmentation_strength,
                        # height_shift_range=self.augmentation_strength,
                        # shear_range=self.augmentation_strength,
                        # horizontal_flip = True,
                        # zoom_range=self.augmentation_strength
                        )

        # no need for augmentation on validation images
        validation_datagen = ImageDataGenerator(
                            preprocessing_function=self.preprocessing
                            )

        holdout_datagen = ImageDataGenerator(
                            preprocessing_function=self.preprocessing
                            )

        self.train_generator = train_datagen.flow_from_directory(
                                        self.train_folder,
                                        target_size=self.target_size,
                                        batch_size=self.batch_size,
                                        class_mode='categorical',
                                        shuffle=True)

        self.validation_generator = validation_datagen.flow_from_directory(
                                                    self.validation_folder,
                                                    target_size=self.target_size,
                                                    batch_size=self.batch_size,
                                                    class_mode='categorical',
                                                    shuffle=False)

        self.holdout_generator = holdout_datagen.flow_from_directory(
                                                self.holdout_folder,
                                                target_size=self.target_size,
                                                batch_size=self.batch_size,
                                                class_mode='categorical',
                                                shuffle=False)


    def fitting(self):
        self.conv()
        self.make_generator()

        sgd = optimizers.SGD(lr=0.005, decay=1e-6, momentum=0.9, nesterov=True)
        self.model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

        early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience=7, verbose=1, mode='auto')

        checkpointer = callbacks.ModelCheckpoint(filepath=('checkpoint.hdf5'), verbose=1, save_best_only=True)

        # balance classes
        counter = Counter(self.train_generator.classes)
        max_val = float(max(counter.values()))
        class_weights = {class_id : max_val/num_images for class_id, num_images in counter.items()}
        pdb.set_trace()
        hist = callbacks.History()

        self.history = self.model.fit_generator(self.train_generator,
                                        steps_per_epoch=self.nTrain/self.batch_size,
                                        epochs=self.nb_epoch,
                                        validation_data=self.validation_generator,
                                        validation_steps=self.nVal/self.batch_size,
                                        callbacks=[checkpointer, early_stopping, hist],
                                        class_weight = class_weights,
                                        workers=15
                                        )

    def evaluate(self):

        self.best_model = load_model('checkpoint.hdf5')
        pdb.set_trace()
        predict = self.best_model.predict_generator(self.holdout_generator,
                                                steps = self.nHoldout/self.batch_size,
                                                use_multiprocessing=True,
                                                verbose=1,
                                                )


        metric = self.best_model.evaluate_generator(self.holdout_generator,
                                            steps=self.nHoldout/self.batch_size,
                                            use_multiprocessing=True,
                                            verbose=1,
                                            )

        self.predicted_class_indices = np.argmax(predict,axis=1)
        pdb.set_trace()
        self.target_names = (self.train_generator.class_indices)
        self.target_names = dict((v,k) for k,v in self.target_names.items())
        self.predictions = [self.target_names[k] for k in self.predicted_class_indices]
        self.filenames=self.holdout_generator.filenames
        self.true_class_indices = self.holdout_generator.classes
        self.true = [self.target_names[k] for k in self.true_class_indices]
        # self.results=pd.DataFrame({"Filename":self.filenames,
        #                       "Predictions":self.predictions})
        pdb.set_trace()
        return metric


    def class_report(self):
        class_names = list(self.target_names.values())
        report = classification_report(self.true_class_indices, self.predicted_class_indices, target_names=class_names)
        cm = confusion_matrix(y_true=self.true_class_indices, y_pred=self.predicted_class_indices)
        pdb.set_trace()
        return class_names, report, cm

    def plot_history(self):
        # Plot training & validation accuracy values
        plt.plot(self.history.history['acc'])
        plt.plot(self.history.history['val_acc'])
        plt.title('Model accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')
        plt.savefig('images/acc_hist.png')
        plt.close()
        # plt.show()

        # Plot training & validation loss values
        plt.plot(self.history.history['loss'])
        plt.plot(self.history.history['val_loss'])
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')
        plt.savefig('images/loss_hist.png')
        plt.close()
        # plt.show()

    def plot_images(self, n=1, image_ind=None):
        '''
        Pulls images with labels and predictions. Can choose specific images with image_ind
        or keep as none to grab n number of plots
        Input: results
                n_plots (int)
                image_ind (list)
        Output: images
        '''
        # Either chooses n number of images or takes in specified images numbers given by image_ind
        if image_ind == None:
            image = np.linspace(0,n-1,n,dtype = int)
        else:
            image = image_ind

        # plot each image in image_ind and save
        for i in image:
            im = plt.imread('data/holdout/'+self.filenames[i])
            plt.imshow(im)
            plt.title(self.true[i])
            plt.xlabel(self.predictions[i])
            plt.savefig('images/image{}'.format(i)+'.png')
            plt.show()


if __name__ == '__main__':
    train_folder = 'data/train'
    validation_folder = 'data/validation'
    holdout_folder = 'data/holdout'
    target_size = (299,299)  # 299,299 is suggested for xception
    CNN = CNNModel(train_folder, validation_folder, holdout_folder, target_size = target_size, preprocessing=preprocess_input)
    CNN.fitting()
    metric = CNN.evaluate()
    class_names, report, cm = CNN.class_report()

    # pickle metrics

    with open('evaluate/metric.txt', 'w') as f:
        f.write(repr(metric))

    with open('evaluate/cm.txt', 'wb') as f2:
        f2.write(repr(cm))

    with open('evaluate/class_names.pkl', 'wb') as f3:
        pickle.dump(class_names, f3)

    with open('evaluate/cr.txt', 'w') as f4:
        f4.write(repr(report))

    # get plots
    CNN.plot_history()
    CNN.plot_images(image_ind = [3,4,5,10,20,51])
