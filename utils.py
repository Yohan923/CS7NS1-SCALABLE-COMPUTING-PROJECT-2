import tensorflow.keras as keras
import tensorflow as tf
import os

tf.__version__

from consts import symbol_map


def to_valid_char(character):
    return symbol_map.get(character) if symbol_map.get(character) else character


def find_actual_char(character):
    if character.isnumeric():
        return ''
    
    for key, value in symbol_map.items():
        if value == character:
            return key

    return character


class ConvertToTfLite(keras.callbacks.Callback):
    def __init__(self, model_dir=None, model_name='model'):
        super(ConvertToTfLite, self).__init__()
        self.model_dir = model_dir
        self.model_name = model_name

    def on_epoch_end(self, epoch, logs=None):
        # Convert the model
        converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
        converter.experimental_new_converter = True
        tflite_model = converter.convert()

        # Save the model.
        with open(self.model_name + '.tflite', 'wb') as f:
            f.write(tflite_model)
            print('model saved in tflite')