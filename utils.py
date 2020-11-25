import tensorflow.keras as keras
import tensorflow as tf
import os

from matplotlib import pyplot as plt

tf.__version__

from consts import symbol_map


def to_valid_char(character):
    return symbol_map.get(character) if symbol_map.get(character) else character


def decode_label_char(character):
    if character.isnumeric():
        return ''
    
    for key, value in symbol_map.items():
        if value == character:
            return key

    return character

def plot_ram(ram_data):
    x = []
    y = []

    for data in ram_data:
        x.append(data[0])
        y.append(data[1]/1000000.0)
    
    plt.plot(x, y, 'bo')
    plt.title('RAM usage across epochs')
    plt.xlabel('epoch')
    plt.ylabel('RAM usage in mega-bytes')

    plt.show()



class ConvertToTfLite(keras.callbacks.Callback):
    def __init__(self, model_dir=None, model_name='model'):
        super(ConvertToTfLite, self).__init__()
        self.model_dir = model_dir
        self.model_name = model_name

    def on_train_end(self, logs=None):
        # Convert the model
        converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
        converter.experimental_new_converter = True
        tflite_model = converter.convert()

        # Save the model.
        with open(self.model_name + '.tflite', 'wb') as f:
            f.write(tflite_model)
            print('model saved in tflite')


class RecordRam(keras.callbacks.Callback):
    def __init__(self, ram_data=None, heap=None):
        super(RecordRam, self).__init__()
        self.ram_data = ram_data
        self.heap = heap

    def on_epoch_end(self, epoch, logs=None):
        x = self.heap.heap()
        self.ram_data.append([epoch + 1, x.size])