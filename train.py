# Import libraries
import tensorflow as tf
import numpy as np
import datetime
from tqdm import tqdm
import os


def create_model():
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=(15, 8, 8)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(256, activation='linear'),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dense(1, activation='linear')
    ])
    model.compile(optimizer=tf.optimizers.legacy.Adam(learning_rate=.002), loss=tf.keras.losses.mean_absolute_error,
                  metrics=['mae'])
    return model


train_dataset = tf.data.Dataset.load("train_datasetv40", compression='GZIP')
print(train_dataset.snapshot)
val_dataset = tf.data.Dataset.load("train_datasetv40", compression='GZIP')

train_dataset = train_dataset.shuffle(buffer_size=10000).batch(512)
val_dataset = val_dataset.batch(512)

if os.path.exists("test 444.h5"):
    model = tf.keras.models.load_model("test 4444.h5")
else:
    model = create_model()

model.summary()

logdir = "logs/loss/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
writer = tf.summary.create_file_writer(logdir)


def log_loss(batch, logs):
    with writer.as_default():
        loss = logs['loss']

        tf.summary.scalar('loss', loss, step=batch)


loss_callback = tf.keras.callbacks.LambdaCallback(on_batch_end=log_loss)

checkpoint_callback = tf.keras.callbacks.ModelCheckpoint('test {epoch}.h5', period=1)

model.fit(train_dataset, validation_data=val_dataset, validation_freq=1, epochs=2000, callbacks=[loss_callback, checkpoint_callback])

model.save('test.h5')
