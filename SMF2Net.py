import tensorflow as tf
from tensorflow.keras import layers
from Module import *


class mymodel(tf.keras.Model):
    def __init__(self,
                 channels = 31 ,
                 vit_num=4,
                 MSSF_num = 4
                 ):
        super(mymodel, self).__init__()

        self.channels = channels
        self.UP = upmodle(channels=self.channels)
        self.vitmode = []
        for _ in range(vit_num):
            self.vitmode.append(Hypervitmode(channels = self.channels, head_dim=40,num_heads=4)) 
        self.SFSPM = SFSPM(channels = self.channels )

        self.MSSF = []
        for _ in range(MSSF_num):
            self.MSSF.append(MFIFB(channels=self.channels, strides = (1, 1), dilation_rate=2, padding = 'same'))

        self.conv3_31 = tf.keras.Sequential([
            layers.Conv2D(self.channels, 1, strides=1, padding='same'),
            layers.BatchNormalization(),
            layers.LeakyReLU(alpha=0.2)
        ])
        self.resblock = []
        for _ in range(3):
            self.resblock.append(ResidualBlock(out_channels=channels))
            
    def call(self, inputs, training=True):
        (Y,Z) = inputs
        ZUP = self.UP(Z)
        y1 = self.conv3_31(Y)
        y2 = tf.add(y1, ZUP)
        res_fe = y2
        for resblock in self.resblock:
            res_fe = resblock(res_fe)
        y3 = tf.add(self.SFSPM(y2), res_fe)
        for spalayer in self.MSSF:
            y2 = spalayer(y2)
        y4 = tf.add(y3, y2)
        for layer in self.vitmode:
            ZUP = layer(ZUP)  
        out = tf.add(y4, ZUP)
        return out


class upmodle(tf.keras.Model):
    def __init__(self,
                 channels,
                 ):
        super(upmodle, self).__init__()

        self.channels = channels
        self.convup32 = tf.keras.Sequential([
            layers.Conv2DTranspose(filters=channels, kernel_size=2, strides=2,
                                   padding='valid', activation=layers.LeakyReLU(alpha=0.2)),
            layers.Conv2DTranspose(filters=channels, kernel_size=2, strides=2,
                                   padding='valid', activation=layers.LeakyReLU(alpha=0.2)),
            layers.Conv2DTranspose(filters=channels, kernel_size=2, strides=2,
                                   padding='valid', activation=layers.LeakyReLU(alpha=0.2)),
            layers.Conv2DTranspose(filters=channels, kernel_size=2, strides=2,
                                   padding='valid', activation=layers.LeakyReLU(alpha=0.2)),
            layers.Conv2DTranspose(filters=channels, kernel_size=2, strides=2,
                                   padding='valid', activation=layers.LeakyReLU(alpha=0.2))
        ])
    def call(self, inputs):
        _, H, W, _ = inputs.shape
        upX_bicubic = tf.image.resize(inputs, (H * 32, W * 32), method="bicubic")
        convup = self.convup32(inputs)
        out = layers.Add()([upX_bicubic, convup])
        return out


class ResidualBlock(tf.keras.Model):
    def __init__(self, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = layers.Conv2D(out_channels, kernel_size=3, strides=stride, padding='same', use_bias=False)
        self.bn1 = layers.BatchNormalization()
        self.relu = layers.ReLU()
        self.conv2 = layers.Conv2D(out_channels, kernel_size=3, strides=stride, padding='same', use_bias=False)
        self.bn2 = layers.BatchNormalization()
        self.shortcut = tf.keras.Sequential()

    def call(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += self.shortcut(identity)
        out = self.relu(out)
        return out

