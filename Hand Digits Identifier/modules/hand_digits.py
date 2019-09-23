import numpy as np
import os
import glob
import shutil
import zipfile
from pathlib import Path
import tensorflow as tf
from keras import backend as K
from keras.utils import np_utils, to_categorical
from keras.layers import Dense, Dropout, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Activation
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array, array_to_img
from keras.models import Sequential, load_model
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt


class sign_lang(object):
    """

    A module to
    1. unzip the .zip file of data
    2. create sub-directories for data generator
    3. move images into corresponding folders (train, validation and test)
    4. create the data generators for training and validation purposes
    5. build a multi-layer CNN model
    6. train the model on training images
    7. evaluate the model on validation images
    8. and perform predictions on test images

    """
    # global variable initializer
    IMAGE_SIZE = 100
    NUM_CLASS = 9
    BATCH_SIZE = 8
    EPOCHS = 20
    target_size = (IMAGE_SIZE, IMAGE_SIZE)
    input_shape = (IMAGE_SIZE, IMAGE_SIZE, 3)
    cwd = os.getcwd()

    def __init__(self, filename):
        self.filename = filename
    # method to get required path

    def get_dir(self, filename=''):
        return self.cwd + '/' + filename
    # method to unzip a .zip file

    def _unzip(self, path):
        filename = path.split('/')[-1]
        folder = str(Path(path).parent) + '/' + filename.split('.')[0]
        if not os.path.exists(folder):
            zfile = zipfile.ZipFile(path, 'r')
            zfile.extractall(folder)
            zfile.close()
            print('Files unzipped to folder: {0}'.format(folder))
        return folder
    # method to make a given directory

    def _make_dir(self, path):
        if not os.path.exists(path):
            os.mkdir(path)
    # method to move files to a target directory

    def _move_files(self, files, path):
        for file in files:
            if not os.path.isfile(path + '/' + file.split('/')[-1]):
                shutil.move(file, path)
    # method to create sub folders (train, val, test) for data generator

    def _prepare_data(self):
        zipfile = self.get_dir(self.filename)
        zipfolder = self._unzip(zipfile)
        data_dir = zipfolder + '/' + zipfolder.split('/')[-1] + '/Dataset/'

        train_dir = data_dir + 'train_set/'
        train_dir_list = [train_dir + repr(i) + '/' for i in range(1, 10)]
        test_dir = data_dir + 'test_set/'
        test_dir_list = [test_dir + repr(i) + '/' for i in range(1, 10)]
        val_dir = data_dir + 'val_set/'
        val_dir_list = [val_dir + repr(i) + '/' for i in range(1, 10)]
        sample_dir = [train_dir, val_dir, test_dir]

        for dir_ in sample_dir:
            self._make_dir(dir_)
            sub_dirs = [dir_ + repr(i) + '/' for i in range(1, 10)]
            for sub_dir in sub_dirs:
                self._make_dir(sub_dir)

        img_dict = {}
        num_train = 150
        num_val = 50
        for i in range(1, 10):
            img_dir = data_dir + repr(i) + '/'
            images = glob.glob(img_dir + '*.JPG')
            img_dict[i] = images
            train_imgs = images[:num_train]
            val_imgs = images[num_train:num_train + num_val]
            test_imgs = images[num_train + num_val:]
            imgs = [train_imgs, val_imgs, test_imgs]
            for img, dir_ in zip(imgs, sample_dir):
                self._move_files(img, dir_ + repr(i) + '/')
        return train_dir, val_dir, test_dir
    # method to build data generators

    def _data_generator(self):
        train_datagen = ImageDataGenerator(rescale=1. / 255,
                                           shear_range=0.4,
                                           zoom_range=0.4,
                                           rotation_range=20,
                                           featurewise_center=True,
                                           featurewise_std_normalization=True
                                           )
        val_datagen = ImageDataGenerator(rescale=1. / 255)
        test_datagen = ImageDataGenerator(rescale=1. / 255)
        train_dir, val_dir, test_dir = self._prepare_data()
        train_generator = train_datagen.flow_from_directory(train_dir,
                                                            target_size=self.target_size,
                                                            batch_size=self.BATCH_SIZE
                                                            )
        val_generator = val_datagen.flow_from_directory(val_dir,
                                                        target_size=self.target_size,
                                                        batch_size=self.BATCH_SIZE,
                                                        )
        test_generator = test_datagen.flow_from_directory(test_dir,
                                                          target_size=self.target_size,
                                                          batch_size=self.BATCH_SIZE,
                                                          )
        return train_generator, val_generator, test_generator
    # method to run (train) the model

    def _run_model(self, model_name):
        train_generator, val_generator, test_generator = self._data_generator()
        earlyStopping = EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='min')
        mcp_save = ModelCheckpoint('hand_digits.h5', save_best_only=True, monitor='val_loss', mode='min')
        callbacks = [earlyStopping, mcp_save]
        model = self._build_model(self.input_shape, self.NUM_CLASS, self.BATCH_SIZE, self.EPOCHS)
        model.fit_generator(train_generator,
                            steps_per_epoch=len(train_generator.filenames) // self.BATCH_SIZE,
                            epochs=self.EPOCHS,
                            callbacks=callbacks,
                            validation_data=val_generator,
                            validation_steps=len(val_generator.filenames) // self.BATCH_SIZE,
                            verbose=1)
        model_dir = self.get_dir('models/')
        model.save(model_dir + model_name)
    # method to build a multi-layer CNN model

    def _build_model(self, input_shape, num_class, batch_size, epochs):
        model = Sequential()
        model.add(Conv2D(32, kernel_size=(3, 3), input_shape=input_shape, activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(units=128, activation='relu'))
        model.add(Dropout(0.25))
        model.add(Dense(units=num_class, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
        return model
    # method to train and save model

    def train(self, model_name):
        if os.path.isfile(self.get_dir('models') + '/' + model_name):
            val = input('Model Exists. Run and overwrite the existed model(Y/N)?')
            if val.lower() == 'y':
                self._run_model(model_name)
            elif val.lower() == 'n':
                print('Keep Existed Model')
        else:
            self._run_model(model_name)
    # method to evaluate model

    def evaluate(self, model_name):
        model_dir = self.get_dir('models/')
        model = load_model(model_dir + model_name)
        _, val_generator, test_generator = self._data_generator()
        STEP_SIZE_TEST = test_generator.n // test_generator.batch_size
        STEP_SIZE_VAL = val_generator.n // val_generator.batch_size
        _, acc = model.evaluate_generator(val_generator, steps=STEP_SIZE_VAL, verbose=0)
        print('Model Accuracy: {0:.1f}%'.format(acc * 100))
    # method to make predictions on any given image

    def predict(self, image):
        img = load_img(image)
        img_data = img_to_array(img).reshape(1, self.IMAGE_SIZE, self.IMAGE_SIZE, 3)
        model_name = 'hand_digits.h5'
        model_dir = self.get_dir('models/')
        model = load_model(model_dir + model_name)
        label = model.predict(img_data)
        label = list(range(1, 10))[np.argmax(label)]
        return label

    def plot_images(self):
        train_dir, _, _ = self._prepare_data()
        train_dir_list = [train_dir + repr(i) + '/' for i in range(1, 10)]
        fig = plt.figure(figsize=(6, 6))
        plt.axis('off')
        for i in range(len(train_dir_list)):
            ax = fig.add_subplot(3, 3, i + 1)
            plt.imshow(plt.imread(glob.glob(train_dir_list[i] + '*.JPG')[0]))
            plt.xticks([])
            plt.yticks([])
            plt.title('Digit: {0}'.format(i + 1), fontweight='bold')
        plt.tight_layout(pad=0, h_pad=0, w_pad=0)
        plt.show()


if __name__ == '__main__':
    filename = 'Sign-Language-Digits-Dataset-master.zip'
    slc = sign_lang(filename)
    slc.train('hand_digits.h5')
    slc.evaluate('hand_digits.h5')
    slc.plot_images()
