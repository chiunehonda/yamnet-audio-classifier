import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_io as tfio

print("TensorFlow version:", tf.__version__)



# load the YAMNet model from TensorFlow Hub, this is where we can take the same features such as motors, buzzing etc for background noise for our drone.

yamnet_model_handle = "https://tfhub.dev/google/yamnet/1"
yamnet_model = hub.load(yamnet_model_handle)

print("YAMNet loaded successfully")


# download one cat recording, this is where we load an unseen drone recording

testing_wav_file_name = tf.keras.utils.get_file('miaow_16k.wav',
                                                'https://storage.googleapis.com/audioset/miaow_16k.wav',
                                                cache_dir='./',
                                                cache_subdir='test_data')

print(testing_wav_file_name)

# Utility functions for loading audio files and making sure the sample rate is correct.

@tf.function
def load_wav_16k_mono(filename):
    """ Load a WAV file, convert it to a float tensor, resample to 16 kHz single-channel audio. """
    file_contents = tf.io.read_file(filename)
    wav, sample_rate = tf.audio.decode_wav(
          file_contents,
          desired_channels=1)
    wav = tf.squeeze(wav, axis=-1)
    sample_rate = tf.cast(sample_rate, dtype=tf.int64)
    wav = tfio.audio.resample(wav, rate_in=sample_rate, rate_out=16000)
    return wav

testing_wav_data = load_wav_16k_mono(testing_wav_file_name)

_ = plt.plot(testing_wav_data)

class_map_path = yamnet_model.class_map_path().numpy().decode('utf-8')
class_names =list(pd.read_csv(class_map_path)['display_name'])

for name in class_names[:20]:
  print(name)
print('...')

scores, embeddings, spectrogram = yamnet_model(testing_wav_data)
class_scores = tf.reduce_mean(scores, axis=0)
top_class = tf.math.argmax(class_scores)
inferred_class = class_names[top_class]

print(f'The main sound is: {inferred_class}')
print(f'The embeddings shape: {embeddings.shape}')

_ = tf.keras.utils.get_file('esc-50.zip',
                        'https://github.com/karoldvl/ESC-50/archive/master.zip',
                        cache_dir='./',
                        cache_subdir='datasets',
                        extract=True)

esc50_csv = './datasets/ESC-50-master/meta/esc50.csv'
base_data_path = './datasets/ESC-50-master/audio/'

pd_data = pd.read_csv(esc50_csv)
pd_data.head()


# create classes

my_classes = ['dog', 'cat']
map_class_to_id = {'dog':0, 'cat':1}

filtered_pd = pd_data[pd_data.category.isin(my_classes)]

class_id = filtered_pd['category'].apply(lambda name: map_class_to_id[name])
filtered_pd = filtered_pd.assign(target=class_id)

full_path = filtered_pd['filename'].apply(lambda row: os.path.join(base_data_path, row))
filtered_pd = filtered_pd.assign(filename=full_path)

# Display in a table assigning the dog nd cat 
print(filtered_pd.head(10))

# Select the two categories for our custom classifier.
my_classes = ["dog", "cat"]

# Convert category names into numerical labels.
map_class_to_id = {
    "dog": 0,
    "cat": 1,
}

# Keep only dog and cat rows from ESC-50.
filtered_pd = pd_data[pd_data.category.isin(my_classes)].copy()

# Assign 0 to dogs and 1 to cats.
class_id = filtered_pd["category"].apply(
    lambda name: map_class_to_id[name]
)

filtered_pd = filtered_pd.assign(target=class_id)

# Turn each short filename into its complete file location.
full_path = filtered_pd["filename"].apply(
    lambda filename: os.path.join(base_data_path, filename)
)

filtered_pd = filtered_pd.assign(filename=full_path)

# Display ten rows in the terminal.
print("\nSelected cat and dog recordings:")
print(filtered_pd[["filename", "fold", "category", "target"]].head(10))

print("\nNumber of recordings per category:")
print(filtered_pd["category"].value_counts())

# Collect the file locations, correct labels and ESC-50 folds.
filenames = filtered_pd["filename"]
targets = filtered_pd["target"]
folds = filtered_pd["fold"]

print("\nNumber of selected audio files:", len(filenames))

# Create a TensorFlow dataset from the table columns.
main_ds = tf.data.Dataset.from_tensor_slices(
    (filenames, targets, folds)
)

print("\nInitial dataset structure:")
print(main_ds.element_spec)

# Replace each filename with its loaded audio waveform.
def load_wav_for_map(filename, label, fold):
    waveform = load_wav_16k_mono(filename)
    return waveform, label, fold


main_ds = main_ds.map(load_wav_for_map)

print("\nDataset structure after loading audio:")
print(main_ds.element_spec)

# Convert one audio waveform into YAMNet embeddings.
def extract_embedding(wav_data, label, fold):
    scores, embeddings, spectrogram = yamnet_model(wav_data)

    # Determine how many audio-frame embeddings YAMNet produced.
    number_of_embeddings = tf.shape(embeddings)[0]

    # Give every embedding the recording's correct label and fold.
    repeated_labels = tf.repeat(label, number_of_embeddings)
    repeated_folds = tf.repeat(fold, number_of_embeddings)

    return embeddings, repeated_labels, repeated_folds


# Apply embedding extraction to every recording.
main_ds = main_ds.map(extract_embedding)

# Turn batches of frame embeddings into individual dataset examples.
main_ds = main_ds.unbatch()

print("\nDataset structure after extracting embeddings:")
print(main_ds.element_spec)

print("\nOriginal recording counts by category and fold:")
print(
    pd.crosstab(
        filtered_pd["category"],
        filtered_pd["fold"],
    )
)

# Cache the embeddings after they are calculated.
cached_ds = main_ds.cache()

# Folds 1, 2 and 3 are used for training.
train_ds = cached_ds.filter(
    lambda embedding, label, fold: fold < 4
)

# Fold 4 is used for validation during training.
val_ds = cached_ds.filter(
    lambda embedding, label, fold: fold == 4
)

# Fold 5 is kept unseen until final evaluation.
test_ds = cached_ds.filter(
    lambda embedding, label, fold: fold == 5
)

# The model needs the embedding and label, but not the fold number.
def remove_fold_column(embedding, label, fold):
    return embedding, label


train_ds = train_ds.map(remove_fold_column)
val_ds = val_ds.map(remove_fold_column)
test_ds = test_ds.map(remove_fold_column)

# Prepare the datasets for efficient model training.
train_ds = (
    train_ds
    .cache()
    .shuffle(1000)
    .batch(32)
    .prefetch(tf.data.AUTOTUNE)
)

val_ds = (
    val_ds
    .cache()
    .batch(32)
    .prefetch(tf.data.AUTOTUNE)
)

test_ds = (
    test_ds
    .cache()
    .batch(32)
    .prefetch(tf.data.AUTOTUNE)
)

print("\nTraining, validation and test pipelines are ready.")

# Create the trainable cat/dog classifier.
my_model = tf.keras.Sequential([
    tf.keras.layers.Input(
        shape=(1024,),
        dtype=tf.float32,
        name="input_embedding",
    ),

    tf.keras.layers.Dense(
        512,
        activation="relu",
        name="hidden_layer",
    ),

    tf.keras.layers.Dense(
        len(my_classes),
        name="class_scores",
    ),
], name="cat_dog_classifier")

print("\nCustom classifier:")
my_model.summary()

# Configure how the model will learn.
my_model.compile(
    loss=tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True
    ),
    optimizer="adam",
    metrics=["accuracy"],
)

# Stop if training stops improving.
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True,
)

print("\nStarting classifier training...")

history = my_model.fit(
    train_ds,
    epochs=20,
    validation_data=val_ds,
    callbacks=[early_stopping],
)

print("\nEvaluating on the unseen test dataset...")

test_loss, test_accuracy = my_model.evaluate(
    test_ds,
    verbose=1,
)

print(f"\nTest loss: {test_loss:.4f}")
print(f"Test accuracy: {test_accuracy:.2%}")

print("\nTraining finished.")

print("\nTesting the classifier on the sample meow recording...")

# Generate YAMNet embeddings for the complete sample recording.
scores, embeddings, spectrogram = yamnet_model(
    testing_wav_data
)

# Produce dog/cat scores for every audio frame.
frame_predictions = my_model(
    embeddings,
    training=False,
).numpy()

# Average the frame scores into one result for the recording.
average_prediction = frame_predictions.mean(axis=0)

# Convert raw scores into probabilities.
probabilities = tf.nn.softmax(
    average_prediction
).numpy()

# Find the category with the largest probability.
predicted_index = int(probabilities.argmax())
predicted_class = my_classes[predicted_index]

print(f"Predicted class: {predicted_class}")

for class_name, probability in zip(
    my_classes,
    probabilities,
):
    print(f"{class_name}: {probability:.2%}")

class ReduceMeanLayer(tf.keras.layers.Layer):
    """Average all frame predictions into one recording prediction."""

    def __init__(self, axis=0, **kwargs):
        super().__init__(**kwargs)
        self.axis = axis

    def call(self, inputs):
        return tf.math.reduce_mean(
            inputs,
            axis=self.axis,
        )

saved_model_path = "./dogs_and_cats_yamnet"

print("\nBuilding the combined waveform model...")

# Input: a one-dimensional 16 kHz audio waveform.
audio_input = tf.keras.layers.Input(
    shape=(),
    dtype=tf.float32,
    name="audio",
)

# Include the pretrained YAMNet inside the new model.
yamnet_layer = hub.KerasLayer(
    yamnet_model_handle,
    trainable=False,
    name="yamnet",
)

# YAMNet generates scores, embeddings and a spectrogram.
yamnet_scores, yamnet_embeddings, yamnet_spectrogram = yamnet_layer(
    audio_input
)

# Pass YAMNet's embeddings through our trained classifier.
frame_class_scores = my_model(
    yamnet_embeddings
)

# Average all frame scores into one result.
recording_class_scores = ReduceMeanLayer(
    axis=0,
    name="classifier",
)(frame_class_scores)

# Connect the waveform input to the final class scores.
serving_model = tf.keras.Model(
    inputs=audio_input,
    outputs=recording_class_scores,
    name="complete_cat_dog_audio_model",
)

print("\nCombined model created.")
serving_model.summary()

print(f"\nSaving model to: {saved_model_path}")

serving_model.save(
    saved_model_path,
    include_optimizer=False,
)

print("Model saved successfully.")

print("\nReloading the saved model...")

reloaded_model = tf.saved_model.load(
    saved_model_path
)

print("Saved model reloaded successfully.")

print("\nTesting the reloaded model on the sample meow...")

reloaded_scores = reloaded_model(
    testing_wav_data
)

reloaded_probabilities = tf.nn.softmax(
    reloaded_scores
).numpy()

predicted_index = int(
    reloaded_probabilities.argmax()
)

predicted_class = my_classes[predicted_index]

print(f"Reloaded model prediction: {predicted_class}")

for class_name, probability in zip(
    my_classes,
    reloaded_probabilities,
):
    print(f"{class_name}: {probability:.2%}")