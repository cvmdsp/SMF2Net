import tensorflow as tf
from tensorflow.keras import layers ,initializers
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import math
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import scipy.io as sio


class SFSPM(tf.keras.Model):
    def __init__(self,
                 channels  ,
                 kernel_size = 3,
                 strides = (1, 1),
                 padding = 'same'
                 ):
        super(SFSPM, self).__init__()
        self.midchannel = channels*2
        self.channels = channels
        self.conv_layer1 = tf.keras.Sequential([
            layers.Conv2D(self.channels, kernel_size, strides=strides,
                          padding=padding),
            layers.BatchNormalization(),
            layers.LeakyReLU(alpha=0.2)
        ])
        self.conv_layer2 = tf.keras.Sequential([
            layers.Conv2D(self.channels, kernel_size, strides=strides,
                          padding=padding),
            layers.BatchNormalization(),
            layers.LeakyReLU(alpha=0.2)
        ])
        self.conv_layer3 = tf.keras.Sequential([
            layers.Conv2D(self.channels, kernel_size, strides=strides,
                          padding=padding),
            layers.BatchNormalization(),
            layers.LeakyReLU(alpha=0.2)
        ])
        self.conv_layer4 = tf.keras.Sequential([
            layers.Conv2D(self.channels, kernel_size, strides=strides,
                          padding=padding),
            layers.BatchNormalization(),
            layers.LeakyReLU(alpha=0.2)
        ])
        self.conv_layer5 = tf.keras.Sequential([
            layers.Conv2D(self.channels, kernel_size, strides=strides,
                          padding=padding),
            layers.BatchNormalization(),
            layers.LeakyReLU(alpha=0.2)
        ])
        self.conv_layer3D = layers.Conv3D(
            filters=1,
            kernel_size=(self.channels, 1, 1),
            strides=(1, 1, 1),
            padding='valid',
            activation=layers.LeakyReLU(alpha=0.2)
        )

    def call(self, input):
        _, weith, high, channel = input.shape
        mapnum = 0
        while mapnum == 0:
            vec_norm = tf.expand_dims(tf.math.l2_normalize(input, axis=-1), axis=1)
            vec_norm = tf.transpose(vec_norm, perm=[0, 4, 2, 3, 1])
            cos = self.conv_layer3D(vec_norm)
            cos = tf.squeeze(cos, axis=1)
            mapo = tf.reduce_sum(cos, axis=-1)
            map_d = 1 - mapo
            map = tf.expand_dims(mapo, axis=-1)
            map_d = tf.expand_dims(map_d, axis=-1)
            mapnum = 1

        input_1 = map* input
        input_2 = map_d* input

        f1 = self.conv_layer1 (input_1)
        f2 = self.conv_layer2 (input_2)

        f_12 = tf.concat([f1, input_2], axis = 3)
        f_21 = tf.concat([f2, input_1], axis=3)

        out1 = self.conv_layer3(f_12)
        out2 = self.conv_layer4(f_21)

        out = tf.concat([out1, out2], axis=3)
        out = self.conv_layer5(out)

        return out


class MFIFB(tf.keras.Model):
    def __init__(self,
                 channels,
                 strides,
                 dilation_rate,
                 padding
                 ):
        super(MFIFB, self).__init__()
        self.channels = channels
        self.strides = strides
        self.dilation_rate = dilation_rate
        self.padding = padding
        self.w0_conv = tf.keras.Sequential([
            layers.Conv2D(self.channels, 1, strides=self.strides, padding=self.padding),
            layers.BatchNormalization(),
            layers.LeakyReLU(alpha=0.2)
        ])
        self.w1_conv = tf.keras.Sequential([
            layers.Conv2D(self.channels, 3, strides=self.strides, padding=self.padding),
            layers.BatchNormalization(),
            layers.LeakyReLU(alpha=0.2)
        ])
        self.w2_conv = tf.keras.Sequential([
            layers.Conv2D(self.channels, 3, strides=self.strides,
                          dilation_rate = self.dilation_rate,padding=self.padding),
            layers.BatchNormalization(),
            layers.LeakyReLU(alpha=0.2)
        ])
        self.w3_conv = tf.keras.Sequential([
            layers.Conv2D(self.channels, 5, strides=self.strides,
                          dilation_rate = 3, padding=self.padding),
            layers.BatchNormalization(),
            layers.LeakyReLU(alpha=0.2)
        ])

        self.conv1_1 = tf.keras.Sequential([
            layers.Conv2D(self.channels, 1, strides=self.strides,
                          dilation_rate=self.dilation_rate, padding=self.padding),
            layers.BatchNormalization(),
            layers.LeakyReLU(alpha=0.2)
        ])
        self.conv1_2 = tf.keras.Sequential([
            layers.Conv2D(self.channels, 1, strides=self.strides,
                          dilation_rate=self.dilation_rate, padding=self.padding),
            layers.BatchNormalization(),
            layers.LeakyReLU(alpha=0.2)
        ])
        self.conv1_3 = tf.keras.Sequential([
            layers.Conv2D(self.channels, 1, strides=self.strides,
                          dilation_rate=self.dilation_rate, padding=self.padding),
            layers.BatchNormalization(),
            layers.LeakyReLU(alpha=0.2)
        ])

        self.conv2_1 = tf.keras.Sequential([
            layers.Conv2D(self.channels, 1, strides=self.strides,
                          dilation_rate=self.dilation_rate, padding=self.padding),
            layers.BatchNormalization(),
            layers.LeakyReLU(alpha=0.2)
        ])


    def call(self, input):
        w0 = self.w0_conv(input)
        w1 = self.w1_conv(input)
        w2 = self.w2_conv(input)
        w3 = self.w3_conv(input)
        w01 = tf.concat([w0,w1], axis=-1)
        w12 = tf.concat([w1, w2], axis=-1)
        w23 = tf.concat([w2, w3], axis=-1)
        L0 = self.conv1_1(w01)
        L1 = self.conv1_2(w12)
        L2 = self.conv1_3(w23)
        L = tf.concat([L0, L1, L2], axis=-1)
        out = self.conv2_1(L)
        return out


class vitEncoder(tf.keras.Model):
    def __init__(self,
                 head_dim,
                 num_heads,
                 outchann
                 ):
        super(vitEncoder, self).__init__()
        self.head_dim= head_dim
        self.num_heads = num_heads
        self.outchann = outchann
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.MHA = MMHA(dim_head=self.head_dim, heads_num=self.num_heads, outc=self.outchann)
        self.layernorm4 = layers.LayerNormalization(epsilon=1e-6)
        self.mlp = mlp(hidden_units=self.outchann,
                       dropout_rate=0.1)

    def call(self, fearure):

        fearure = self.layernorm1(fearure)
        x2 = self.MHA(fearure)
        x3 = layers.Add()([x2, fearure])
        x4 = self.layernorm4(x3)
        out = self.mlp(x4)
        out = layers.Add()([x3, out])
        return out


class Hypervitmode(tf.keras.Model):
    def __init__(self,
                 channels,
                 head_dim,
                 num_heads
                 ):
        super(Hypervitmode, self).__init__()
        self.head_dim= head_dim
        self.num_heads = num_heads
        self.channels = channels
        self.vitEncoder = vitEncoder(head_dim = self.head_dim,num_heads = self.num_heads,outchann=self.channels)

    def call(self, fearure):

        b = tf.shape(fearure)[0]
        H = tf.shape(fearure)[1]
        W = tf.shape(fearure)[2]
        C = tf.shape(fearure)[3]

        x = self.vitEncoder(fearure)
        output = tf.reshape(x, [b, H, W, C])
        return output


class MMHA(tf.keras.Model):
    def __init__(self,
                 dim_head,
                 heads_num,
                 outc
                 ):
        super(MMHA, self).__init__()
        self.dim_head = dim_head
        self.heads_num = heads_num
        self.outc = outc

        self.fc1 = tf.keras.layers.Dense(self.dim_head * self.heads_num,  use_bias=False)
        self.fc2 = tf.keras.layers.Dense(self.dim_head * self.heads_num, use_bias=False)
        self.fc3 = tf.keras.layers.Dense(self.dim_head * self.heads_num, use_bias=False)

        self.mlp = mlp(hidden_units=self.outc, dropout_rate=0.1)
        self.avg = layers.GlobalAveragePooling2D()
        self.max = layers.GlobalMaxPooling2D()

        self.fcout = tf.keras.layers.Dense(self.outc, use_bias=False)

    def call(self, fearure):
        _ = tf.shape(fearure)[0]
        dim_h = tf.shape(fearure)[1]
        dim_w = tf.shape(fearure)[2]
        dim_c = tf.shape(fearure)[3]

        spe_max = self.max(fearure)
        spe_avg = self.avg(fearure)
        spe_max = self.mlp(spe_max)
        spe_avg = self.mlp(spe_avg)
        spe_atten = tf.sigmoid(tf.add(spe_max, spe_avg))

        fearure = tf.reshape(fearure, [_, dim_h * dim_w, dim_c])
        q1_inp = self.fc1(fearure)
        k1_inp = self.fc2(fearure)
        v1_inp = self.fc3(fearure)

        q_spa = tf.transpose(tf.reshape(q1_inp, [_, dim_h * dim_w, self.heads_num, self.dim_head]),
                         perm=[0, 2, 1, 3])
        k_spa = tf.transpose(tf.reshape(k1_inp, [_, dim_h * dim_w, self.heads_num, self.dim_head]), perm=[0, 2, 1, 3])
        v_spa = tf.transpose(tf.reshape(v1_inp, [_, dim_h * dim_w, self.heads_num, self.dim_head]), perm=[0, 2, 1, 3])

        q_spa = tf.transpose(q_spa, perm=[0, 1, 3, 2])
        k_spa = tf.transpose(k_spa, perm=[0, 1, 3, 2])
        v_spa = tf.transpose(v_spa, perm=[0, 1, 3, 2])

        q_spa = tf.linalg.l2_normalize(q_spa, axis=-1)
        k_spa = tf.linalg.l2_normalize(k_spa, axis=-1)
        q_spa = tf.transpose(q_spa, perm=[0, 1, 3, 2])
        attn = k_spa @ q_spa
        attn = tf.nn.softmax(attn, axis=-1)
        x_spa = attn @ v_spa

        x = tf.transpose(x_spa, perm=[0, 1, 3, 2])
        x = tf.reshape(x, [_, dim_h * dim_w, self.heads_num * self.dim_head])
        x = tf.reshape(self.fcout(x), [_, dim_h, dim_w, dim_c])
        x = tf.multiply(x, tf.reshape(spe_atten, [_, 1, 1, dim_c]))

        return x

class mlp(tf.keras.Model):
    def __init__(self,
                 hidden_units,
                 dropout_rate
                 ):
        super(mlp, self).__init__()
        self.hidden_units1 = hidden_units*2
        self.hidden_units2 = hidden_units
        self.dropout_rate = dropout_rate
        self.dense1 = layers.Dense(self.hidden_units1, activation='relu')
        self.dense2 = layers.Dense(self.hidden_units2, activation='relu')
        self.drop1 = layers.Dropout(self.dropout_rate)
        self.drop2 = layers.Dropout(self.dropout_rate)

    def call(self, inputs):

        x = self.dense1(inputs)
        x_d = self.drop1(x)
        x1 = self.dense2(x_d)
        x1_d = self.drop2(x1)
        return x1_d
