# NEWT (Hayes et al. 2021) core idea: a *shared, pointwise* MLP waveshaper
# applied to many parallel copies of the exciter, each wrapped by an affine
# (scale/shift) pre/post transform whose params come from the controls (FiLM).
#   X      : (B, S=512, 4)   -> [:, :, 0] core triangle, [:, :, 1:4] control CVs
#   y_true : (B, S, 1)       -> target morphed wave

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

IN_D = 4
OUT_D = 1


def build_newt(num_waveshapers=64, ctrl_hidden=128, shaper_hidden=32, shaper_depth=2):

    x = keras.Input(shape=(None, IN_D), name="x")

    # triangle input (B, S, 1)
    exciter = x[..., 0:1]
    # control CVs (B, S, 3)
    ctrl = x[..., 1:4]

    # control -> FiLM params:
    # pre (scale,shift) + post (scale,shift) per shaper
    h = layers.Dense(ctrl_hidden, activation="relu")(ctrl)
    h = layers.Dense(ctrl_hidden, activation="relu")(h)
    film = layers.Dense(num_waveshapers * 4)(h)  # (B, S, 4N)
    a_pre, b_pre, a_post, b_post = tf.split(film, 4, axis=-1)

    # shared pointwise waveshaper
    shaper = keras.Sequential(name="waveshaper")
    for _ in range(shaper_depth):
        shaper.add(layers.Dense(shaper_hidden, activation="tanh"))
    shaper.add(layers.Dense(1, activation="tanh"))

    # broadcast exciter to N shapers; then affine -> shared shaper -> affine
    e = tf.tile(exciter, [1, 1, num_waveshapers])  # (B, S, N)
    z = a_pre * e + b_pre
    z = shaper(z[..., tf.newaxis])[..., 0]  # pointwise over (B, S, N)
    z = a_post * z + b_post

    # mixer
    y = layers.Dense(OUT_D, activation="tanh", name="mix")(z)  # (B, S, 1)
    return keras.Model(x, y, name="newt")


# def train(model, ds, val_ds=None, epochs=200, lr=1e-3):
#     model.compile(
#         optimizer=keras.optimizers.Adam(lr),
#         # loss=keras.losses.Huber(),
#         loss=keras.losses.Huber(),
#         metrics=["mae"],
#     )
#     return model.fit(ds, validation_data=val_ds, epochs=epochs)


# if __name__ == "__main__":
#     model = build_newt()
#     model.summary()
