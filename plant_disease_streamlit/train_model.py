"""Train the CNN using the PlantVillage dataset and export files for the Streamlit app."""
import json
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

SEED = 42
IMG_SIZE = (128, 128)
BATCH_SIZE = 32
EPOCHS = 2


def main():
    tf.keras.utils.set_random_seed(SEED)
    print("Loading PlantVillage dataset (first run downloads the dataset)...")
    ds_all, ds_info = tfds.load(
        "plant_village", split="train", as_supervised=True,
        with_info=True, shuffle_files=True
    )
    class_names = ds_info.features["label"].names
    num_classes = len(class_names)
    total_count = int(ds_info.splits["train"].num_examples)

    train_count = int(total_count * 0.70)
    val_count = int(total_count * 0.15)
    shuffled = ds_all.shuffle(
        buffer_size=min(total_count, 10000), seed=SEED,
        reshuffle_each_iteration=False
    )
    train_raw = shuffled.take(train_count)
    remaining = shuffled.skip(train_count)
    val_raw = remaining.take(val_count)

    def preprocess(image, label):
        image = tf.image.resize(image, IMG_SIZE)
        image = tf.cast(image, tf.float32) / 255.0
        return image, tf.cast(label, tf.int32)

    autotune = tf.data.AUTOTUNE
    train_ds = train_raw.map(preprocess, num_parallel_calls=autotune).batch(BATCH_SIZE).prefetch(autotune)
    val_ds = val_raw.map(preprocess, num_parallel_calls=autotune).batch(BATCH_SIZE).prefetch(autotune)

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3)),
        tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu"),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu"),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(128, 3, padding="same", activation="relu"),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dropout(0.30),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(0.20),
        tf.keras.layers.Dense(num_classes, activation="softmax"),
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy", metrics=["accuracy"]
    )
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint(
            "best_plant_leaf_cnn.keras", monitor="val_loss", save_best_only=True
        ),
    ]
    model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, callbacks=callbacks)
    model.save("plant_leaf_disease_cnn.keras")
    with open("class_names.json", "w", encoding="utf-8") as file:
        json.dump(class_names, file, indent=2)
    print("Created plant_leaf_disease_cnn.keras and class_names.json")


if __name__ == "__main__":
    main()
