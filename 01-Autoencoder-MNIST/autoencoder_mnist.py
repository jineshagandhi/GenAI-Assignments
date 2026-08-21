import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models

(x_train, _), (x_test, _) = tf.keras.datasets.mnist.load_data()

print("Training data shape:", x_train.shape)
print("Testing data shape:", x_test.shape)

x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

x_train = np.expand_dims(x_train, axis=-1)
x_test = np.expand_dims(x_test, axis=-1)

print("After preprocessing:")
print("Training data shape:", x_train.shape)
print("Testing data shape:", x_test.shape)

encoder = models.Sequential([
    layers.Input(shape=(28, 28, 1)),
    layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
    layers.MaxPooling2D((2, 2), padding="same"),
    layers.Conv2D(16, (3, 3), activation="relu", padding="same"),
    layers.MaxPooling2D((2, 2), padding="same")
], name="Encoder")

decoder = models.Sequential([
    layers.Conv2D(16, (3, 3), activation="relu", padding="same"),
    layers.UpSampling2D((2, 2)),
    layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
    layers.UpSampling2D((2, 2)),
    layers.Conv2D(1, (3, 3), activation="sigmoid", padding="same")
], name="Decoder")

input_image = layers.Input(shape=(28, 28, 1))

encoded = encoder(input_image)
decoded = decoder(encoded)

autoencoder = models.Model(
    input_image,
    decoded,
    name="Convolutional_Autoencoder"
)

print("\nEncoder Architecture:")
encoder.summary()

print("\nDecoder Architecture:")
decoder.summary()

print("\nAutoencoder Architecture:")
autoencoder.summary()

autoencoder.compile(
    optimizer="adam",
    loss="binary_crossentropy"
)

history = autoencoder.fit(
    x_train,
    x_train,
    epochs=10,
    batch_size=128,
    shuffle=True,
    validation_data=(x_test, x_test)
)

decoded_images = autoencoder.predict(x_test)

n = 10

plt.figure(figsize=(20, 4))

for i in range(n):
    plt.subplot(2, n, i + 1)
    plt.imshow(x_test[i].reshape(28, 28), cmap="gray")
    plt.axis("off")

    plt.subplot(2, n, i + 1 + n)
    plt.imshow(decoded_images[i].reshape(28, 28), cmap="gray")
    plt.axis("off")

plt.suptitle("Original Images vs Reconstructed Images")
plt.tight_layout()
plt.savefig("outputs/reconstructed_images.png")
plt.show()

plt.figure(figsize=(8, 5))

plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Autoencoder Training Loss")
plt.legend()

plt.savefig("outputs/training_loss.png")
plt.show()

autoencoder.save("outputs/mnist_autoencoder.keras")

print("\nTraining completed successfully.")
print("Results saved in the outputs folder.")