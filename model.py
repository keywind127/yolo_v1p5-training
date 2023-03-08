# YOLO Model

from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.models import Sequential, Model 
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import (
    BatchNormalization,
    AveragePooling2D,
    MaxPooling2D,
    Concatenate,
    LeakyReLU,
    Softmax,
    Flatten,
    Dropout,
    Reshape,
    Conv2D,
    Input,
    Dense
)
from typing import *

class YoloModel:  
    architecture_convolution = [
        (64, 7, 2, 3),
        "M",
        (192, 3, 1, 1),
        "M",
        (128, 1, 1, 0),
        (256, 3, 1, 1),
        (256, 1, 1, 0),
        (512, 3, 1, 1),
        "M",
        [ 4, (256, 1, 1, 1), (512, 3, 1, 1) ],
        (512, 1, 1, 0),
        (1024, 3, 1, 1),
        "M",
        [ 2, (512, 1, 1, 0), (1024, 3, 1, 1) ],
        (1024, 3, 1, 1),
        (1024, 3, 2, 1),
        (1024, 3, 1, 1),
        (1024, 3, 1, 1)
    ]
    @classmethod 
    def _build_convolution_darknet(class_, input_shape : Tuple[ int, int, int ]) -> Model:
        input_layer = Input(input_shape)
        output_layer = input_layer 
        for new_layer in class_.architecture_convolution:
            if (isinstance(new_layer, str)):
                output_layer = MaxPooling2D((2, 2))(output_layer)
                continue 
            if (isinstance(new_layer, tuple)):
                new_layer = [ 1, new_layer ]
            (quantity, conv_list) = (new_layer[0], new_layer[1:])
            for _ in range(quantity):
                for conv_layer in conv_list:
                    (filters, kernel_size, strides, padding) = conv_layer
                    output_layer = Conv2D(filters, (kernel_size, kernel_size), strides = (strides, strides), padding = "same", use_bias = False)(output_layer)
                    output_layer = BatchNormalization()(output_layer)
                    output_layer = LeakyReLU(0.1)(output_layer)
        return Model(input_layer, output_layer)

    @classmethod 
    def _build_convolution_vgg19(class_, input_shape : Tuple[ int, int, int ]) -> Model:
        input_layer = Input(input_shape)
        output_layer = input_layer 
        output_layer = VGG19(weights = "imagenet", include_top = False, input_tensor = output_layer)
        for layer in output_layer.layers:
            layer.trainable = False
        output_layer = output_layer.output

        output_layer = Conv2D(1024, (1, 1), strides = (1, 1), padding = "same", use_bias = False)(output_layer)
        output_layer = BatchNormalization()(output_layer)
        output_layer = LeakyReLU(0.1)(output_layer)

        output_layer = Conv2D(1024, (3, 3), strides = (1, 1), padding = "same", use_bias = False)(output_layer)
        output_layer = BatchNormalization()(output_layer)
        output_layer = LeakyReLU(0.1)(output_layer)

        output_layer = Conv2D(1024, (1, 1), strides = (1, 1), padding = "same", use_bias = False)(output_layer)
        output_layer = BatchNormalization()(output_layer)
        output_layer = LeakyReLU(0.1)(output_layer)

        output_layer = Conv2D(1024, (3, 3), strides = (2, 2), padding = "same", use_bias = False)(output_layer)
        output_layer = BatchNormalization()(output_layer)
        output_layer = LeakyReLU(0.1)(output_layer)

        return Model(input_layer, output_layer)

    @classmethod 
    def build_model(class_, S : int, C : int, input_shape : Tuple[ int, int, int ], use_vgg19 : bool = True) -> Model:
        assert isinstance(S, int)
        assert isinstance(C, int)
        assert isinstance(input_shape, tuple)
        assert isinstance(use_vgg19, bool)
        convolutions = ((class_._build_convolution_vgg19(input_shape)) if (use_vgg19) else (class_._build_convolution_darknet(input_shape)))
        output_layer = convolutions.output
        output_layer = Flatten()(output_layer)
        output_layer = Dense(512)(output_layer)
        output_layer = Dropout(0.10)(output_layer)
        output_layer = LeakyReLU(0.1)(output_layer)
        output_layer_1 = Dense(S * S * C, activation = "linear")(output_layer)
        output_layer_1 = Reshape((S, S, C))(output_layer_1)
        output_layer_1 = Softmax(axis = 3)(output_layer_1)
        output_layer_2 = Dense(S * S * 1, activation = "sigmoid")(output_layer)
        output_layer_2 = Reshape((S, S, 1))(output_layer_2)
        output_layer_3 = Dense(S * S * 4, activation = "linear")(output_layer)
        output_layer_3 = Reshape((S, S, 4))(output_layer_3)
        output_layer_4 = Dense(S * S * 1, activation = "sigmoid")(output_layer)
        output_layer_4 = Reshape((S, S, 1))(output_layer_4)
        output_layer_5 = Dense(S * S * 4, activation = "linear")(output_layer)
        output_layer_5 = Reshape((S, S, 4))(output_layer_5)
        output_layer = Concatenate(axis = 3)([ output_layer_1, output_layer_2, output_layer_3, output_layer_4, output_layer_5 ])
        return Model(convolutions.input, output_layer) 