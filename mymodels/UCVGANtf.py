#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@author: 杨亚峰<yang_armageddon@163.com>
@version: 1.0.0
@license:  Apache Licence
@editor: Pycharm yyf
@file: UCVGANtf.py
@datatime: 8/14/2023 1:17 AM
"""
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

class BasicBlock(tf.keras.layers.Layer):
    def __init__(self, out_channels):
        super().__init__()
        self.norm = layers.LayerNormalization()
        self.conv = layers.Conv2D(out_channels, kernel_size=3, padding='same')
        self.activation = layers.LeakyReLU()

    def call(self, x):
        x = self.norm(x)
        x = self.conv(x)
        return self.activation(x)

class DownBlock(tf.keras.layers.Layer):
    def __init__(self, out_channels):
        super().__init__()
        self.block = models.Sequential([
            BasicBlock(out_channels),
            BasicBlock(out_channels),
            layers.Conv2D(out_channels, kernel_size=2, strides=2)
        ])

    def call(self, x):
        return self.block(x)

class UpBlock(tf.keras.layers.Layer):
    def __init__(self, out_channels):
        super().__init__()
        self.up_sample = models.Sequential([
            layers.UpSampling2D(size=2, interpolation='bilinear'),
            layers.Conv2D(out_channels, kernel_size=3, padding='same')
        ])
        self.block = models.Sequential([
            BasicBlock(out_channels * 2),
            BasicBlock(out_channels)
        ])

    def call(self, x, skip_connection):
        x = self.up_sample(x)
        x = tf.concat([x, skip_connection], axis=-1)
        return self.block(x)

class GELU(tf.keras.layers.Layer):
    def call(self, x):
        return 0.5 * x * (1 + tf.math.tanh(tf.math.sqrt(2 / np.pi) * (x + 0.044715 * tf.pow(x, 3))))

class PositionWiseFFN(tf.keras.layers.Layer):
    def __init__(self, features, ffn_features, activ='gelu'):
        super().__init__()
        self.net = tf.keras.Sequential([
            tf.keras.layers.Dense(ffn_features),
            GELU() if activ == 'gelu' else tf.keras.layers.ReLU(),
            tf.keras.layers.Dense(features),
        ])

    def call(self, x):
        return self.net(x)


class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, features, ffn_features, n_heads, activ='gelu', rezero=True):
        super().__init__()
        self.norm1 = layers.LayerNormalization()
        self.atten = layers.MultiHeadAttention(num_heads=n_heads, key_dim=features)
        self.norm2 = layers.LayerNormalization()
        self.ffn = PositionWiseFFN(features, ffn_features, activ)
        self.rezero = rezero
        if rezero:
            self.re_alpha = tf.Variable(initial_value=0., trainable=True)

    def call(self, x):
        y1 = self.norm1(x)
        y1 = self.atten(y1, y1, y1)
        y = x + self.re_alpha * y1 if self.rezero else x + y1
        y2 = self.norm2(y)
        y2 = self.ffn(y2)
        y = y + self.re_alpha * y2 if self.rezero else y + y2
        return y


class TransformerEncoder(tf.keras.layers.Layer):
    def __init__(self, features, ffn_features, n_heads, n_blocks, activ, rezero=True):
        super().__init__()
        self.encoder = [TransformerBlock(features, ffn_features, n_heads, activ, rezero) for _ in range(n_blocks)]

    def call(self, x):
        y = x
        for block in self.encoder:
            y = block(y)
        return y


class FourierEmbedding(tf.keras.layers.Layer):
    def __init__(self, features, height, width):
        super().__init__()
        self.projector = layers.Dense(features)
        self._height = height
        self._width = width

    def call(self, y, x):
        x_norm = 2 * x / (self._width - 1) - 1
        y_norm = 2 * y / (self._height - 1) - 1
        z = tf.concat([x_norm[..., None], y_norm[..., None]], axis=-1)
        return tf.math.sin(self.projector(z))


class ViTInput(tf.keras.layers.Layer):
    def __init__(self, input_features, embed_features, features, height, width):
        super().__init__()
        self._height = height
        self._width = width
        self.embed = FourierEmbedding(embed_features, height, width)
        self.out = layers.Dense(features)

    def call(self, x):
        y, x = tf.meshgrid(tf.range(self._height), tf.range(self._width), indexing="ij")
        y = tf.cast(y, tf.float32)
        x = tf.cast(x, tf.float32)
        embed = self.embed(y, x)
        embed = tf.broadcast_to(embed[None, ...], [tf.shape(x)[0]] + list(embed.shape))
        result = tf.concat([embed, x], axis=-1)
        return self.out(result)


class PixelwiseViT(tf.keras.layers.Layer):
    def __init__(self, features, n_heads, n_blocks, ffn_features, embed_features, activ, image_shape, rezero=True):
        super().__init__()
        self.image_shape = image_shape
        self.trans_input = ViTInput(image_shape[0], embed_features, features, image_shape[1], image_shape[2])
        self.encoder = TransformerEncoder(features, ffn_features, n_heads, n_blocks, activ, rezero)
        self.trans_output = layers.Dense(image_shape[0])

    def call(self, x):
        itokens = tf.reshape(x, [-1, self.image_shape[1] * self.image_shape[2], self.image_shape[0]])
        y = self.trans_input(itokens)
        y = self.encoder(y)
        otokens = self.trans_output(y)
        result = tf.reshape(otokens, [-1, self.image_shape[1], self.image_shape[2], self.image_shape[0]])
        return result


class UNet(models.Model):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.down1 = DownBlock(64)
        self.down2 = DownBlock(128)
        self.down3 = DownBlock(256)
        self.down4 = DownBlock(512)
        self.bottleneck = PixelwiseViT(512, 8, 12, 2048, 512, 'gelu', (512, 256 // 16, 256 // 16))
        self.up1 = UpBlock(256)
        self.up2 = UpBlock(128)
        self.up3 = UpBlock(64)
        self.up4 = UpBlock(out_channels)

    def call(self, x):
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        bottleneck = self.bottleneck(d4)
        u1 = self.up1(bottleneck, d3)
        u2 = self.up2(u1, d2)
        u3 = self.up3(u2, d1)
        return self.up4(u3, x)


if __name__ == '__main__':
    model = UNet(3, 3)
    model.build((None, 256, 256, 3))
    model.summary()
    tf.keras.utils.plot_model(model, show_shapes=True)



