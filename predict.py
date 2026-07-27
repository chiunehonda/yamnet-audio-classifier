import sys

import numpy as np
import tensorflow as tf
import tensorflow_io as tfio


MODEL_PATH = "./dogs_and_cats_yamnet"
CLASS_NAMES = ["dog", "cat"]


def load_wav_16k_mono(filename):
    """Load a WAV file and convert it to mono 16 kHz audio."""

    file_contents = tf.io.read_file(filename)

    waveform, sample_rate = tf.audio.decode_wav(
        file_contents,
        desired_channels=1,
    )

    waveform = tf.squeeze(waveform, axis=-1)
    sample_rate = tf.cast(sample_rate, tf.int64)

    waveform = tfio.audio.resample(
        waveform,
        rate_in=sample_rate,
        rate_out=16000,
    )

    return waveform


def main():
    if len(sys.argv) != 2:
        print("Usage: python predict.py path_to_audio.wav")
        return

    audio_path = sys.argv[1]

    print(f"Loading model from: {MODEL_PATH}")
    model = tf.saved_model.load(MODEL_PATH)

    print(f"Loading audio: {audio_path}")
    waveform = load_wav_16k_mono(audio_path)

    raw_scores = model(waveform)
    probabilities = tf.nn.softmax(raw_scores).numpy()

    predicted_index = int(np.argmax(probabilities))
    predicted_class = CLASS_NAMES[predicted_index]

    print("\nPrediction results:")

    for class_name, probability in zip(
        CLASS_NAMES,
        probabilities,
    ):
        print(f"{class_name}: {probability:.2%}")

    print(f"\nPredicted class: {predicted_class}")


if __name__ == "__main__":
    main()