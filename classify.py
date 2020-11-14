#!/usr/bin/env python3

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

import os
import cv2
import numpy
import string
import random
import argparse
import tensorflow as tf
import tensorflow.keras as keras
#import tflite_runtime.interpreter as tflite
import scipy.ndimage

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

def decode(characters, y):
    y = numpy.argmax(numpy.array(y), axis=2)[:,0]
    return ''.join([characters[x] for x in y])

# A utility function to decode the output of the network
def decode_batch_predictions(characters, pred):
    input_len = numpy.ones(pred.shape[0]) * pred.shape[1]
    # Use greedy search. For complex tasks, you can use beam search
    results = keras.backend.ctc_decode(pred, input_length=input_len, greedy=True)[0][0][:, :50]
    # Iterate over the results and get back the text
    output_text = []
    for res in results:
        res = "".join([characters[c] for c in res])
        output_text.append(res)
    return "".join(output_text)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-name', help='Model name to use for classification', type=str)
    parser.add_argument('--captcha-dir', help='Where to read the captchas to break', type=str)
    parser.add_argument('--output', help='File where the classifications should be saved', type=str)
    parser.add_argument('--symbols', help='File with the symbols to use in captchas', type=str)
    parser.add_argument('--tflite', help='if we are clasifying using tflite', type=str)

    args = parser.parse_args()

    if args.model_name is None:
        print("Please specify the CNN model to use")
        exit(1)

    if args.captcha_dir is None:
        print("Please specify the directory with captchas to break")
        exit(1)

    if args.output is None:
        print("Please specify the path to the output file")
        exit(1)

    if args.symbols is None:
        print("Please specify the captcha symbols file")
        exit(1)

    symbols_file = open(args.symbols, 'r')
    captcha_symbols = symbols_file.readline().strip()
    captcha_symbols = [ch for ch in captcha_symbols]
    captcha_symbols.append('')
    symbols_file.close()

    # with tf.device('/cpu:0'):
    with open(args.output, 'w', newline='\n') as output_file:

        if args.tflite:
            interpreter = tf.lite.Interpreter(model_path=args.model_name+'.tflite')
            interpreter.allocate_tensors()

            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()

            input_shape = input_details[1]['shape']

        else:
            model = tf.keras.models.load_model('model.h5', custom_objects={'CTCLayer': CTCLayer})
            model.load_weights(args.model_name+'.h5')

            prediction_model = keras.models.Model(
                model.get_layer(name="images").input, model.get_layer(name="softmax").output
            )
            prediction_model.summary()

        for x in os.listdir(args.captcha_dir):
            
            #raw_image = cv2.imread(os.path.join(args.captcha_dir, x))
            ## to grayscale
            #gray = cv2.cvtColor(raw_image, cv2.COLOR_BGR2GRAY)
            ## thresholding
            #ret, thresh = cv2.threshold(gray, 230, 255, cv2.THRESH_BINARY)
            #thresh = ~thresh
#
            ## erosion to reduce noise
            #kernel = numpy.ones((2, 2),numpy.uint8)
            #erosion = cv2.erode(thresh,kernel,iterations = 1)
            #erosion = ~erosion # black letters, white background
#
            #img = scipy.ndimage.median_filter(erosion, (5, 1)) # remove line noise
            #img = scipy.ndimage.median_filter(img, (1, 3)) # weaken circle noise
#
            ## img = cv2.erode(img, kernel, iterations = 1) # dilate image to initial stage (erode works similar to dilate because we thresholded the image the opposite way)
            #
            #img = scipy.ndimage.median_filter(img, (3, 3)) # remove any final 'weak' noise that might be present (line or circle)
            #
            #res = cv2.resize(img,(64, 128), interpolation = cv2.INTER_LINEAR)
#
            #img = numpy.array(img) / 255.0
            #img = numpy.reshape(img, (64, 128, 1))
#
            #img = numpy.transpose(img, (1, 0, 2))
            #img = numpy.reshape(img, (-1, 128, 64, 1))


            # 1. Read image
            img = tf.io.read_file(os.path.join(args.captcha_dir, x))
            # 2. Decode and convert to grayscale
            img = tf.io.decode_png(img, channels=1)
            # 3. Convert to float32 in [0, 1] range
            img = tf.image.convert_image_dtype(img, tf.float32)
            # 4. Resize to the desired size
            img = tf.image.resize(img, [64, 128])
            # 5. Transpose the image because we want the time
            # dimension to correspond to the width of the image.
            img = tf.transpose(img, perm=[1, 0, 2])
            img = tf.reshape(img, (-1, 128, 64, 1))    

            interpreter.set_tensor(input_details[1]['index'], img)

            interpreter.invoke()            

            prediction = interpreter.get_tensor(output_details[0]['index'])
            print(prediction)

            #prediction = prediction_model.predict(img)
            output_file.write(x + "," + decode_batch_predictions(captcha_symbols, prediction) + "\n")

            print('Classified ' + x)

if __name__ == '__main__':
    main()
