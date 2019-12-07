import tensorflow as tf
import numpy as np


# def generator(feature_depth, name, units_list=[128, 256, 256]):
#     model = tf.keras.Sequential(name=name)
#     for units in units_list:
#         model.add(tf.keras.layers.Dense(units=units, activation=tf.nn.leaky_relu))
#     model.add(tf.keras.layers.Dense(units=feature_depth, activation=tf.nn.tanh))

#     return model


# def discriminator(name, units_list=[256, 256, 128]):
#     model = tf.keras.Sequential(name=name)
#     for units in units_list:
#         model.add(tf.keras.layers.Dense(units=units, activation=tf.nn.relu))
#     model.add(tf.keras.layers.Dense(units=1))

#     return model


def generator(project_shape, filters_list, strides_list, name="generator"):
    model = tf.keras.Sequential(name=name)
    model.add(tf.keras.layers.Dense(
        units=np.prod(project_shape),
        use_bias=False,
        kernel_initializer=tf.random_normal_initializer(mean=0., stddev=0.02)
    ))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.ReLU())
    model.add(tf.keras.layers.Reshape(target_shape=project_shape))
    for filters, strides in zip(filters_list[:-1], strides_list[:-1]):
        model.add(tf.keras.layers.Conv2DTranspose(
            filters=filters,
            kernel_size=[5, 5],
            strides=strides,
            padding="same",
            use_bias=False,
            kernel_initializer=tf.random_normal_initializer(mean=0., stddev=0.02)
        ))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.ReLU())
    model.add(tf.keras.layers.Conv2DTranspose(
        filters=filters_list[-1],
        kernel_size=[5, 5],
        strides=strides_list[-1],
        padding="same",
        activation=tf.nn.tanh,
        kernel_initializer=tf.random_normal_initializer(mean=0., stddev=0.02)
    ))

    return model


def discriminator(filters_list, strides_list, name="discriminator"):
    model = tf.keras.Sequential(name=name)
    for filters, strides in zip(filters_list, strides_list):
        model.add(tf.keras.layers.Conv2D(
            filters=filters,
            kernel_size=[5, 5],
            strides=strides,
            padding="same",
            use_bias=False,
            kernel_initializer=tf.random_normal_initializer(mean=0., stddev=0.02)
        ))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.LeakyReLU(alpha=0.2))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(
        units=1,
        activation=tf.nn.sigmoid,
        kernel_initializer=tf.random_normal_initializer(mean=0., stddev=0.02)
    ))

    return model


# class WGAN(object):
#     def __init__(self, feature_depth):
#         self.feature_depth = feature_depth

#         self.generator = generator(self.feature_depth, "generator")
#         self.discriminator = discriminator("discriminator")
class WGAN(object):
    def __init__(
        self,
        project_shape,
        gen_filters_list,
        gen_strides_list,
        disc_filters_list,
        disc_strides_list
    ):
        self.project_shape = project_shape
        self.gen_filters_list = gen_filters_list
        self.gen_strides_list = gen_strides_list
        self.disc_filters_list = disc_filters_list
        self.disc_strides_list = disc_strides_list

        self.generator = generator(self.project_shape, self.gen_filters_list, self.gen_strides_list)
        self.discriminator = discriminator(self.disc_filters_list, self.disc_strides_list)
    
    def generator_loss(self, z):
        x = self.generator(z, training=True)
        fake_score = self.discriminator(x, training=True)

        loss = -tf.reduce_mean(fake_score)

        return loss
    
    def discriminator_loss(self, x, z):
        x_fake = self.generator(z, training=True)
        true_score = self.discriminator(x, training=True)
        fake_score = self.discriminator(x_fake, training=True)

        loss = -tf.reduce_mean(true_score) + tf.reduce_mean(fake_score)
        
        return loss