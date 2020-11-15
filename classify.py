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
import tflite_runtime.interpreter as tflite

# import tensorflow.keras as keras
# import tensorflow as tf

def greedy_ctc_decode(pred):
    tmp = numpy.array(pred)
    result = []
    prev = 51 # can be arbitrary
    for step in pred[0]:
        idx = numpy.argmax(step)
        if idx != prev:
            result.append(idx)
            prev = idx

    return result


# A utility function to decode the output of the network
def decode_batch_predictions(characters, pred):
    # input_len = numpy.ones(pred.shape[0]) * pred.shape[1]
    # # Use greedy search. For complex tasks, you can use beam search
    # results = keras.backend.ctc_decode(pred, input_length=input_len, greedy=True)[0][0][:, :50]
    # # Iterate over the results and get back the text
    # output_text = []
    # for res in results:
    #     res = "".join([characters[c] for c in res])
    #     output_text.append(res)


    results = greedy_ctc_decode(pred)
    output_text = []
    for res in results:
        output_text.append(characters[res])

    return "".join(output_text)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-name', help='Model name to use for classification', type=str)
    parser.add_argument('--captcha-dir', help='Where to read the captchas to break', type=str)
    parser.add_argument('--output', help='File where the classifications should be saved', type=str)
    parser.add_argument('--symbols', help='File with the symbols to use in captchas', type=str)

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

    with open(args.output, 'w', newline='\n') as output_file:
        # interpreter = tf.lite.Interpreter(model_path=args.model_name+'.tflite')

        interpreter = tflite.Interpreter(model_path=args.model_name+'.tflite')
        interpreter.allocate_tensors()

        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        input_shape = input_details[1]['shape']

        for x in os.listdir(args.captcha_dir):
            
            img = cv2.imread(os.path.join(args.captcha_dir, x), cv2.IMREAD_GRAYSCALE)
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

            img = numpy.reshape(img, input_shape)
   
            interpreter.set_tensor(input_details[1]['index'], img)
            interpreter.invoke()            

            prediction = interpreter.get_tensor(output_details[0]['index'])

            output_file.write(x + "," + decode_batch_predictions(captcha_symbols, prediction) + "\n")

            print('Classified ' + x)

if __name__ == '__main__':
    main()
