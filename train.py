#!/usr/bin/env python3

import tensorflow.keras as keras
import tensorflow as tf
import argparse
import random
import string
import numpy
import cv2
import os
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

from utils import decode_label_char, ConvertToTfLite, RecordRam, plot_ram
import time

from guppy import hpy
# import scipy.ndimage

class CTCLayer(keras.layers.Layer):
    def __init__(self, name=None, **kwargs):
        super(CTCLayer, self).__init__(name=name, **kwargs)
        self.loss_fn = keras.backend.ctc_batch_cost

    def call(self, y_true, y_pred):

        # Compute the training-time loss value and add it
        # to the layer using `self.add_loss()`.
        batch_len = tf.cast(tf.shape(y_true)[0], dtype="int64")
        input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")
        label_length = tf.cast(tf.shape(y_true)[1], dtype="int64")

        input_length = input_length * tf.ones(shape=(batch_len, 1), dtype="int64")
        label_length = label_length * tf.ones(shape=(batch_len, 1), dtype="int64")

        loss = self.loss_fn(y_true, y_pred, input_length, label_length)
        self.add_loss(loss)

        # At test time, just return the computed predictions
        return y_pred

    def get_config(self):
        config = super(CTCLayer, self).get_config()
        return config


# Build a Keras model given some parameters

def create_model(captcha_num_symbols, input_shape, model_depth=2, module_size=2):
    input_tensor = keras.layers.Input(name="images", shape=input_shape)
    labels = keras.layers.Input(name="labels", shape=(None,))
    
    # First conv block
    x = keras.layers.Conv2D(
        32,
        (3, 3),
        activation="relu",
        kernel_initializer="he_normal",
        padding="same",
        name="Conv1",
    )(input_tensor)
    x = keras.layers.MaxPooling2D((2, 2), name="pool1")(x)

    # Second conv block
    x = keras.layers.Conv2D(
        64,
        (3, 3),
        activation="relu",
        kernel_initializer="he_normal",
        padding="same",
        name="Conv2",
    )(x)
    x = keras.layers.MaxPooling2D((2, 2), name="pool2")(x)

    # We have used two max pool with pool size and strides 2.
    # Hence, downsampled feature maps are 4x smaller. The number of
    # filters in the last layer is 64. Reshape accordingly before
    # passing the output to the RNN part of the model
    new_shape = ((128 // 4), (64 // 4) * 64)
    x = keras.layers.Reshape(target_shape=new_shape, name="reshape")(x)
    x = keras.layers.Dense(64, activation="relu", name="dense1")(x)
    x = keras.layers.Dropout(0.2)(x)

    # RNNs
    x = keras.layers.Bidirectional(keras.layers.LSTM(128, return_sequences=True, dropout=0.25))(x)
    x = keras.layers.Bidirectional(keras.layers.LSTM(64, return_sequences=True, dropout=0.25))(x)

    # Output layer
    x = keras.layers.Dense(captcha_num_symbols, activation="softmax", name="softmax")(x)

    output = CTCLayer(name="ctc_loss")(labels, x)

    model = keras.models.Model(
        inputs=[input_tensor, labels], outputs=output, name="captcha_solver"
    )

    return model

# A Sequence represents a dataset for training in Keras
# In this case, we have a folder full of images
# Elements of a Sequence are *batches* of images, of some size batch_size


class ImageSequence(keras.utils.Sequence):
    def __init__(self, directory_name, batch_size, captcha_symbols, captcha_width, captcha_height):
        self.directory_name = directory_name
        self.batch_size = batch_size
        self.captcha_symbols = captcha_symbols
        self.captcha_width = captcha_width
        self.captcha_height = captcha_height

        file_list = os.listdir(self.directory_name)
        self.files = dict(
            zip(map(lambda x: x.split('.')[0], file_list), file_list))
        self.used_files = []
        self.count = len(file_list)

    def __len__(self):
        return int(numpy.floor(self.count / self.batch_size)) - 1

    def __getitem__(self, idx):
        X = numpy.zeros((self.batch_size, self.captcha_width, self.captcha_height, 1), dtype=numpy.float32)
        labels = numpy.full([self.batch_size, 16], len(self.captcha_symbols) - 1) # 16 being max number of char in a label, we fill matrix with "" char

        for i in range(self.batch_size):

            random_image_label = random.choice(list(self.files.keys()))
            random_image_file = self.files[random_image_label]

            # We've used this image now, so we can't repeat it in this iteration
            self.used_files.append(self.files.pop(random_image_label))

            #raw_image = cv2.imread(os.path.join(self.directory_name, random_image_file))
            ## to grayscale
            #gray = cv2.cvtColor(raw_image, cv2.COLOR_BGR2GRAY)
            ## thresholding
            #ret, thresh = cv2.threshold(gray, 230, 255, cv2.THRESH_BINARY)
            #thresh = ~thresh

            ## erosion to reduce noise
            #kernel = numpy.ones((2, 2),numpy.uint8)
            #erosion = cv2.erode(thresh,kernel,iterations = 1)
            #erosion = ~erosion # black letters, white background

            #img = scipy.ndimage.median_filter(erosion, (5, 1)) # remove line noise
            #img = scipy.ndimage.median_filter(img, (1, 3)) # weaken circle noise

            ## img = cv2.erode(img, kernel, iterations = 1) # dilate image to initial stage (erode works similar to dilate because we thresholded the image the opposite way)
            #
            #img = scipy.ndimage.median_filter(img, (3, 3)) # remove any final 'weak' noise that might be present (line or circle)
            #
            #res = cv2.resize(img,(self.captcha_height, self.captcha_width), interpolation = cv2.INTER_LINEAR)

            #img = numpy.array(img) / 255.0
            #img = numpy.reshape(img, (self.captcha_height, self.captcha_width, 1))

            #img = numpy.transpose(img, (1, 0, 2))

            # plt.imshow(img,'gray')
            # plt.title('ok')
            # plt.xticks([]),plt.yticks([])

            # plt.show()


            img = cv2.imread(os.path.join(self.directory_name, random_image_file), cv2.IMREAD_GRAYSCALE)
            # to grayscale
            img = numpy.reshape(img, (img.shape[0], img.shape[1], 1))
            # Convert to float32 in [0, 1] range
            img = numpy.array(img, dtype=numpy.float32) / 255.0
            # Resize to the desired size
            img = cv2.resize(img,(128, 64), interpolation = cv2.INTER_LINEAR)
            img = numpy.reshape(img, (img.shape[0], img.shape[1], 1))
            # Transpose the image because we want the time
            # dimension to correspond to the width of the image.
            img = numpy.transpose(img, (1, 0, 2))

            X[i] = img

            # We have a little hack here - we save captchas as TEXT_num.png if there is more than one captcha with the text "TEXT"
            # So the real label should have the "_num" stripped out.
            # each character is separated by "_", illegal characters are mapped by symbol_map and decoded before using
            random_image_label = random_image_label.split('_')
            pos = 0
            for ch in random_image_label:
                decoded_char = decode_label_char(ch)

                if decoded_char: # decoded char may be "" - this corresponds to the version number i.e. "label_2.png"
                    labels[i][pos] = self.captcha_symbols.index(decoded_char)
                    pos += 1

        return {"images": X, "labels": labels}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--width', help='Width of captcha image', type=int)
    parser.add_argument('--height', help='Height of captcha image', type=int)
    parser.add_argument(
        '--batch-size', help='How many images in training captcha batches', type=int)
    parser.add_argument(
        '--train-dataset', help='Where to look for the training image dataset', type=str)
    parser.add_argument(
        '--validate-dataset', help='Where to look for the validation image dataset', type=str)
    parser.add_argument('--output-model-name',
                        help='Where to save the trained model', type=str)
    parser.add_argument(
        '--input-model', help='Where to look for the input model to continue training', type=str)
    parser.add_argument(
        '--epochs', help='How many training epochs to run', type=int)
    parser.add_argument(
        '--symbols', help='File with the symbols to use in captchas', type=str)
    args = parser.parse_args()

    if args.width is None:
        print("Please specify the captcha image width")
        exit(1)

    if args.height is None:
        print("Please specify the captcha image height")
        exit(1)

    if args.batch_size is None:
        print("Please specify the training batch size")
        exit(1)

    if args.epochs is None:
        print("Please specify the number of training epochs to run")
        exit(1)

    if args.train_dataset is None:
        print("Please specify the path to the training data set")
        exit(1)

    if args.validate_dataset is None:
        print("Please specify the path to the validation data set")
        exit(1)

    if args.output_model_name is None:
        print("Please specify a name for the trained model")
        exit(1)

    if args.symbols is None:
        print("Please specify the captcha symbols file")
        exit(1)

    h = hpy()

    ram_data = []

    captcha_symbols = None
    with open(args.symbols) as symbols_file:
        captcha_symbols = symbols_file.readline()
    captcha_symbols = [ch for ch in captcha_symbols]
    captcha_symbols.append('')

    device = '/device:CPU:0'
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    if len(physical_devices) > 0: # "GPU available!"
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
        device = '/device:GPU:0'

    with tf.device(device):
        # with tf.device('/device:XLA_CPU:0'):
        print(f'training with {device}')
        if args.input_model is not None and os.path.exists(args.input_model + '.h5'):
            model = tf.keras.models.load_model(args.input_model + '.h5', custom_objects={'CTCLayer': CTCLayer})
            model.load_weights(args.input_model + '.h5')
        else:
            model = create_model(len(captcha_symbols), (args.width, args.height, 1))
            model.compile(optimizer=keras.optimizers.Adam(1e-3, amsgrad=True))

        model.summary()

        training_data = ImageSequence(
            args.train_dataset, args.batch_size, captcha_symbols, args.width, args.height)
        validation_data = ImageSequence(
            args.validate_dataset, args.batch_size, captcha_symbols, args.width, args.height)


        early_stopping_patience = 10
        # Add early stopping
        early_stopping = keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=early_stopping_patience, restore_best_weights=True, verbose=1
        )
        callbacks = [early_stopping,
                     # keras.callbacks.CSVLogger('log.csv'),
                     keras.callbacks.ModelCheckpoint(args.output_model_name+'.h5', save_best_only=False),
                     ConvertToTfLite(model_dir=args.output_model_name+'.h5', model_name=args.output_model_name),
                     RecordRam(ram_data=ram_data, heap=h)
                     ]

        # Save the model architecture to JSON
        with open(args.output_model_name+".json", "w") as json_file:
            json_file.write(model.to_json())

        try:
            train_start = time.time()
            model.fit_generator(generator=training_data,
                                validation_data=validation_data,
                                epochs=args.epochs,
                                callbacks=callbacks,
                                use_multiprocessing=True)
            train_end = time.time()
            plot_ram(ram_data=ram_data)
            print(f'Total time taken for training = {float(train_end - train_start)} seconds')

        except KeyboardInterrupt:
            print('KeyboardInterrupt caught, saving current weights as ' +
                  args.output_model_name+'_resume.h5')
            model.save_weights(args.output_model_name+'_resume.h5')


if __name__ == '__main__':
    main()
