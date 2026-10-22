"""Train a CNN using the local, class-folder-organized digit dataset."""

from __future__ import annotations

import argparse
from pathlib import Path


CLASS_NAMES = [str(digit) for digit in range(10)]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif"}


def is_image_directory(path: Path) -> bool:
    return path.is_dir() and all(
        (path / class_name).is_dir()
        and any(
            image.is_file() and image.suffix.lower() in IMAGE_EXTENSIONS
            for image in (path / class_name).iterdir()
        )
        for class_name in CLASS_NAMES
    )


def find_training_directory(dataset_dir: Path) -> Path:
    candidates = [
        dataset_dir / "trainingSet" / "trainingSet",
        dataset_dir / "trainingSet",
        dataset_dir / "trainingSample" / "trainingSample",
        dataset_dir / "trainingSample",
        dataset_dir,
    ]
    for candidate in candidates:
        if is_image_directory(candidate):
            return candidate

    raise FileNotFoundError(
        f"Could not find folders named 0 through 9 with images under {dataset_dir}. "
        "Expected a layout such as MNIST Dataset/trainingSet/trainingSet/0/ ... /9/."
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a handwritten digit CNN from local image folders."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "MNIST Dataset",
        help="Dataset root containing trainingSet/trainingSet/0 through /9.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "digit_cnn.keras",
        help="Path for the trained Keras model file.",
    )
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=128)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        raise SystemExit("--epochs and --batch-size must both be positive.")

    try:
        import tensorflow as tf
    except Exception as exc:
        raise SystemExit(
            "TensorFlow could not be imported. Check that the installed TensorFlow "
            "and NumPy versions are compatible."
        ) from exc

    try:
        training_dir = find_training_directory(args.data_dir.expanduser().resolve())
    except FileNotFoundError as exc:
        raise SystemExit(str(exc)) from exc

    print(f"Loading labeled images from: {training_dir}")
    training_data = tf.keras.utils.image_dataset_from_directory(
        training_dir,
        labels="inferred",
        label_mode="int",
        class_names=CLASS_NAMES,
        color_mode="grayscale",
        image_size=(28, 28),
        batch_size=args.batch_size,
        shuffle=True,
        seed=42,
        validation_split=0.15,
        subset="training",
    )
    validation_data = tf.keras.utils.image_dataset_from_directory(
        training_dir,
        labels="inferred",
        label_mode="int",
        class_names=CLASS_NAMES,
        color_mode="grayscale",
        image_size=(28, 28),
        batch_size=args.batch_size,
        shuffle=True,
        seed=42,
        validation_split=0.15,
        subset="validation",
    )

    autotune = tf.data.AUTOTUNE
    training_data = training_data.cache().prefetch(buffer_size=autotune)
    validation_data = validation_data.cache().prefetch(buffer_size=autotune)

    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(28, 28, 1)),
            tf.keras.layers.Rescaling(1.0 / 255),
            tf.keras.layers.Conv2D(32, kernel_size=3, padding="same", activation="relu"),
            tf.keras.layers.MaxPooling2D(pool_size=2),
            tf.keras.layers.Conv2D(64, kernel_size=3, padding="same", activation="relu"),
            tf.keras.layers.MaxPooling2D(pool_size=2),
            tf.keras.layers.Dropout(0.25),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dropout(0.4),
            tf.keras.layers.Dense(10, activation="softmax"),
        ],
        name="handwritten_digit_cnn",
    )
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=3,
            restore_best_weights=True,
        )
    ]
    model.fit(
        training_data,
        validation_data=validation_data,
        epochs=args.epochs,
        callbacks=callbacks,
    )

    output_path = args.output.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(output_path)
    print(f"\nModel saved to: {output_path}")


if __name__ == "__main__":
    main()