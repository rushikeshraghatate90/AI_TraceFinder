# build_cnn.py
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
from tensorflow.keras import layers, models, Input


def conv_block(x, filters, pool=True, dropout_rate=0.25):
    """Basic Conv2D block with BN + Dropout"""
    x = layers.Conv2D(filters, (3, 3), padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(filters, (3, 3), padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    if pool:
        x = layers.MaxPooling2D((2, 2))(x)
    if dropout_rate:
        x = layers.Dropout(dropout_rate)(x)
    return x


def build_dual_branch(input_shape_img=(256, 256, 3), input_shape_noise=(256, 256, 1), num_classes=11):
    """Dual-branch CNN: Image branch + Noise branch"""

    # Branch 1: Image
    img_input = Input(shape=input_shape_img, name="official_input")
    x1 = conv_block(img_input, 32)
    x1 = conv_block(x1, 64)
    x1 = conv_block(x1, 128)
    x1 = layers.GlobalAveragePooling2D()(x1)

    # Branch 2: Noise
    noise_input = Input(shape=input_shape_noise, name="noise_input")
    x2 = conv_block(noise_input, 32)
    x2 = conv_block(x2, 64)
    x2 = conv_block(x2, 128)
    x2 = layers.GlobalAveragePooling2D()(x2)

    # Fusion
    combined = layers.Concatenate()([x1, x2])
    z = layers.Dense(256, activation="relu")(combined)
    z = layers.Dropout(0.5)(z)
    output = layers.Dense(num_classes, activation="softmax")(z)

    # Model
    model = models.Model(inputs=[img_input, noise_input], outputs=output)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss="sparse_categorical_crossentropy",  # use categorical_crossentropy if y is one-hot
        metrics=["accuracy"]
    )

    model.summary()
    return model
