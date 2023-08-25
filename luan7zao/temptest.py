import tensorflow as tf
from tensorflow.keras.layers import Conv2D, ReLU, LeakyReLU, Concatenate, Conv2DTranspose, Input
from tensorflow.keras import Sequential, Model
import numpy as np

# 定义分布式策略
strategy = tf.distribute.MirroredStrategy()

import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) == 0:
    print("No GPUs were found.")
else:
    print("Found the following GPUs:", physical_devices)


# 在策略作用域内定义模型、损失函数和训练步骤
with strategy.scope():
    # 定义下采样块
    def downsample(filters, size):
        initializer = tf.random_normal_initializer(0., 0.02)
        block = Sequential([
            Conv2D(filters, size, strides=2, padding='same', kernel_initializer=initializer, use_bias=False),
            LeakyReLU(),
            tf.keras.layers.BatchNormalization()
        ])
        return block

    # 定义上采样块
    def upsample(filters, size):
        initializer = tf.random_normal_initializer(0., 0.02)
        block = Sequential([
            Conv2DTranspose(filters, size, strides=2, padding='same', kernel_initializer=initializer, use_bias=False),
            ReLU(),
            tf.keras.layers.BatchNormalization()
        ])
        return block

    # 定义生成器
    def Generator():
        inputs = Input(shape=[256, 256, 3])
        down_stack = [downsample(64, 4), downsample(128, 4), downsample(256, 4)]
        up_stack = [upsample(128, 4), upsample(64, 4)]

        initializer = tf.random_normal_initializer(0., 0.02)
        last = Conv2DTranspose(3, 4, strides=2, padding='same', kernel_initializer=initializer, activation='tanh')

        x = inputs
        skips = []
        for down in down_stack:
            x = down(x)
            skips.append(x)

        skips = reversed(skips[:-1])

        for up, skip in zip(up_stack, skips):
            x = up(x)
            x = Concatenate()([x, skip])

        x = last(x)
        return Model(inputs=inputs, outputs=x)

    # 定义判别器
    def Discriminator():
        initializer = tf.random_normal_initializer(0., 0.02)
        inp = Input(shape=[256, 256, 3], name='input_image')
        x = inp

        x = downsample(64, 4)(x)
        x = downsample(128, 4)(x)
        x = downsample(256, 4)(x)

        x = Conv2D(1, 4, strides=1, kernel_initializer=initializer)(x)
        return Model(inputs=inp, outputs=x)

    # 定义损失函数和优化器
    loss_obj = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    def discriminator_loss(disc_real_output, disc_generated_output):
        real_loss = loss_obj(tf.ones_like(disc_real_output), disc_real_output)
        generated_loss = loss_obj(tf.zeros_like(disc_generated_output), disc_generated_output)
        total_disc_loss = real_loss + generated_loss
        return total_disc_loss

    def generator_loss(disc_generated_output, gen_output, target):
        gan_loss = loss_obj(tf.ones_like(disc_generated_output), disc_generated_output)
        l1_loss = tf.reduce_mean(tf.abs(target - gen_output))
        total_gen_loss = gan_loss + (100 * l1_loss)
        return total_gen_loss

    generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
    discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

    # 定义训练步骤
    @tf.function
    def train_step(input_image, target):
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            gen_output = generator(input_image, training=True)

            disc_real_output = discriminator(target, training=True)
            disc_generated_output = discriminator(gen_output, training=True)

            gen_loss = generator_loss(disc_generated_output, gen_output, target)
            disc_loss = discriminator_loss(disc_real_output, disc_generated_output)

            print('gen_loss: ', gen_loss)

        generator_gradients = gen_tape.gradient(gen_loss, generator.trainable_variables)
        discriminator_gradients = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

        generator_optimizer.apply_gradients(zip(generator_gradients, generator.trainable_variables))
        discriminator_optimizer.apply_gradients(zip(discriminator_gradients, discriminator.trainable_variables))

    # 创建生成器和判别器
    generator = Generator()
    discriminator = Discriminator()

# 创建虚拟数据
def generate_fake_data(num_samples=1000):
    return np.random.rand(num_samples, 256, 256, 3).astype('float32')

train_images_A = generate_fake_data()
train_images_B = generate_fake_data()

# 使用tf.data创建数据集
dataset = tf.data.Dataset.from_tensor_slices((train_images_A, train_images_B))
dataset = dataset.batch(32)

# 定义训练循环
def train(dataset, epochs):
    for epoch in range(epochs):
        for input_image, target in dataset:
            train_step(input_image, target)

# 训练CycleGAN模型
train(dataset, epochs=100)
