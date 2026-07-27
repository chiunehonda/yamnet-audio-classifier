# YAMNet Cat and Dog Audio Classifier

This is a Python audio-classification project built with TensorFlow
and Google's pretrained YAMNet model.

I completed this project to learn how audio transfer learning works
before building a custom drone-sound detector.

## What the project does

The program:

1. Downloads the ESC-50 environmental-audio dataset.
2. Selects 40 dog and 40 cat recordings.
3. Converts every recording to mono 16 kHz audio.
4. Uses YAMNet to extract 1,024-dimensional audio embeddings.
5. Splits complete recordings into training, validation and test sets.
6. Trains a custom neural network to classify dog and cat sounds.
7. Evaluates the classifier on unseen recordings.
8. Combines YAMNet and the custom classifier into one saved model.
9. Uses a separate prediction program to classify new WAV files.

## Project structure

- `train.py` prepares the data, trains the classifier, evaluates it
  and exports the finished model.
- `predict.py` loads the exported model and tests a new WAV file.
- `requirements.txt` lists the required Python packages.
- `.gitignore` prevents generated data, models and environments from
  being uploaded.

## Technologies

- Python 3.10
- TensorFlow 2.11
- TensorFlow Hub
- TensorFlow I/O
- YAMNet
- pandas
- NumPy

## Installation

Create a Python 3.10 virtual environment:

```powershell
py -3.10 -m venv .venv
.venv\Scripts\Activate.ps1

## Results

### ESC-50 test dataset

- Test accuracy: 99.83%
- Sample meow prediction: cat

### Independent dog-barking test

I tested the exported model using a dog-barking recording that was not
part of the TensorFlow tutorial or the ESC-50 training dataset.

- Audio source: BigSoundBank, “Barking Dogs,” sound 288
- License: CC0 / public domain
- Predicted class: dog
- Dog score: 99.80%
- Cat score: 0.20%

Source:
https://bigsoundbank.com/barking-dogs-s0288.html