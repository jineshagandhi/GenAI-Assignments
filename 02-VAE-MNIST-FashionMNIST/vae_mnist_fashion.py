import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os


os.makedirs("output", exist_ok=True)


LATENT_DIM = 16
EPOCHS = 10
BATCH_SIZE = 128


class Sampling(layers.Layer):
    def call(self, inputs):
        mean, log_var = inputs
        epsilon = tf.random.normal(shape=tf.shape(mean))
        return mean + tf.exp(0.5 * log_var) * epsilon


class Encoder(keras.Model):
    def __init__(self):
        super().__init__(name="VAE_Encoder")

        self.conv1 = layers.Conv2D(32, 3, padding="same", activation="relu")
        self.pool1 = layers.MaxPooling2D()
        self.conv2 = layers.Conv2D(16, 3, padding="same", activation="relu")
        self.pool2 = layers.MaxPooling2D()
        self.flatten = layers.Flatten()
        self.dense = layers.Dense(64, activation="relu")
        self.mean = layers.Dense(LATENT_DIM)
        self.log_var = layers.Dense(LATENT_DIM)
        self.sampling = Sampling()

    def call(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.flatten(x)
        x = self.dense(x)

        mean = self.mean(x)
        log_var = self.log_var(x)
        z = self.sampling((mean, log_var))

        return mean, log_var, z


class Decoder(keras.Model):
    def __init__(self):
        super().__init__(name="VAE_Decoder")

        self.dense = layers.Dense(7 * 7 * 16, activation="relu")
        self.reshape = layers.Reshape((7, 7, 16))
        self.conv1 = layers.Conv2D(16, 3, padding="same", activation="relu")
        self.up1 = layers.UpSampling2D()
        self.conv2 = layers.Conv2D(32, 3, padding="same", activation="relu")
        self.up2 = layers.UpSampling2D()
        self.output_layer = layers.Conv2D(
            1, 3, padding="same", activation="sigmoid"
        )

    def call(self, z):
        x = self.dense(z)
        x = self.reshape(x)
        x = self.conv1(x)
        x = self.up1(x)
        x = self.conv2(x)
        x = self.up2(x)
        return self.output_layer(x)


class VAE(keras.Model):
    def __init__(self, encoder, decoder):
        super().__init__(name="VAE")

        self.encoder = encoder
        self.decoder = decoder

        self.loss_tracker = keras.metrics.Mean(name="loss")
        self.reconstruction_tracker = keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.kl_tracker = keras.metrics.Mean(name="kl_loss")

    @property
    def metrics(self):
        return [
            self.loss_tracker,
            self.reconstruction_tracker,
            self.kl_tracker
        ]

    def call(self, inputs):
        mean, log_var, z = self.encoder(inputs)
        reconstruction = self.decoder(z)
        return reconstruction

    def train_step(self, data):
        if isinstance(data, tuple):
            data = data[0]

        with tf.GradientTape() as tape:
            mean, log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)

            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(
                    keras.losses.binary_crossentropy(
                        data, reconstruction
                    ),
                    axis=(1, 2)
                )
            )

            kl_loss = -0.5 * tf.reduce_mean(
                tf.reduce_sum(
                    1 + log_var
                    - tf.square(mean)
                    - tf.exp(log_var),
                    axis=1
                )
            )

            total_loss = reconstruction_loss + kl_loss

        gradients = tape.gradient(
            total_loss,
            self.trainable_weights
        )

        self.optimizer.apply_gradients(
            zip(gradients, self.trainable_weights)
        )

        self.loss_tracker.update_state(total_loss)
        self.reconstruction_tracker.update_state(reconstruction_loss)
        self.kl_tracker.update_state(kl_loss)

        return {
            "loss": self.loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_tracker.result(),
            "kl_loss": self.kl_tracker.result()
        }


def load_data(dataset):
    if dataset == "MNIST":
        (x_train, _), (x_test, _) = keras.datasets.mnist.load_data()
    else:
        (x_train, _), (x_test, _) = keras.datasets.fashion_mnist.load_data()

    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)

    return x_train, x_test


def show_reconstruction(vae, x_test, dataset):
    _, _, z = vae.encoder(x_test[:10])
    reconstructed = vae.decoder(z)

    plt.figure(figsize=(10, 4))

    for i in range(10):
        plt.subplot(2, 10, i + 1)
        plt.imshow(x_test[i].squeeze(), cmap="gray")
        plt.axis("off")

        plt.subplot(2, 10, i + 11)
        plt.imshow(
            reconstructed[i].numpy().squeeze(),
            cmap="gray"
        )
        plt.axis("off")

    plt.suptitle(dataset + " - Original and Reconstructed")
    plt.tight_layout()
    plt.savefig(
    f"output/{dataset.lower().replace('-', '_')}_reconstruction.png"
    )
    plt.show()
    plt.close()


def generate_images(vae, dataset):
    random_z = tf.random.normal((10, LATENT_DIM))
    generated = vae.decoder(random_z)

    plt.figure(figsize=(10, 4))

    for i in range(10):
        plt.subplot(2, 5, i + 1)
        plt.imshow(
            generated[i].numpy().squeeze(),
            cmap="gray"
        )
        plt.axis("off")

    plt.suptitle(dataset + " - Generated Images")
    plt.tight_layout()
    plt.savefig(
    f"output/{dataset.lower().replace('-', '_')}_generated.png"
    )
    plt.show()
    plt.close()


def run_experiment(dataset):
    print("\n" + "=" * 50)
    print(dataset + " VAE")
    print("=" * 50)

    x_train, x_test = load_data(dataset)

    print("Training data shape:", x_train.shape)
    print("Testing data shape:", x_test.shape)

    encoder = Encoder()
    decoder = Decoder()
    vae = VAE(encoder, decoder)

    vae.compile(
        optimizer=keras.optimizers.Adam()
    )

    vae(x_train[:1])

    print("\nEncoder Architecture:")
    encoder.summary()

    print("\nDecoder Architecture:")
    decoder.summary()

    history = vae.fit(
        x_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE
    )

    show_reconstruction(
        vae,
        x_test,
        dataset
    )

    generate_images(
        vae,
        dataset
    )

    plt.figure(figsize=(8, 5))
    plt.plot(history.history["loss"])
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(dataset + " VAE Training Loss")
    plt.grid()
    plt.savefig(
        f"output/{dataset.lower().replace('-', '_')}_training_loss.png"
    )
    plt.show()
    plt.close()


run_experiment("MNIST")
run_experiment("Fashion-MNIST")