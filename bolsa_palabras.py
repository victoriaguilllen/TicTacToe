import cv2
import numpy as np
import sys

from utils import *

from bow import BoW
from dataset import Dataset
from image_classifier import ImageClassifier
import time
from tqdm import tqdm
import pickle
import os

script_dir = os.path.dirname(os.path.abspath(__file__))


training_set = Dataset.load(f"{script_dir}/dataset/training", "*jpg")
validation_set = Dataset.load(f"{script_dir}/dataset/validation", "*jpg")


def initialise_classifier():
    bow = BoW()
    bow.load_vocabulary(f"{script_dir}/vocabulary")

    image_classifier = ImageClassifier(bow)
    image_classifier.load("classifier")

    return image_classifier


def extract_descriptors():
    feature_extractor = cv2.SIFT_create()
    descriptors = []
    for path in tqdm(training_set, unit="image", file=sys.stdout):
        descriptor = None
        image = cv2.imread(path, cv2.COLOR_BGR2GRAY)

        try:
            _, descriptor = feature_extractor.detectAndCompute(image, None)
        except:
            print(f"WARN: Issue generating descriptor for image {path}")

        if descriptor is not None:
            descriptors.append(descriptor)
    return descriptors


def create_vocabulary(descriptors):
    # Build vocabulary
    vocabulary_size = 70
    iterations = 20
    termination_criteria = (cv2.TERM_CRITERIA_MAX_ITER | cv2.TERM_CRITERIA_EPS, iterations, 1e-6)
    words = cv2.BOWKMeansTrainer(vocabulary_size, termination_criteria)

    # Add all descriptors
    words.add(np.vstack(descriptors))

    time.sleep(0.1)  # Prevents a race condition between tqdm and print statements.
    print("\nClustering descriptors into", vocabulary_size, "words using K-means...")
    vocabulary = words.cluster()
    filename =  f"{script_dir}/vocabulary.pickle"

    # Open the file from above in the write and binary mode
    with open(filename, "wb") as f:
        pickle.dump(["SIFT", vocabulary], f, pickle.HIGHEST_PROTOCOL)


def training_classificator():
    bow = BoW()

    bow.load_vocabulary(f"{script_dir}/vocabulary")
    image_classifier = ImageClassifier(bow)

    # Args for the training method
    image_classifier.train(training_set)
    classifier = "classifier"
    image_classifier.save(classifier)
    return image_classifier, classifier


def inference(classifier, set):
    bow = BoW()
    bow.load_vocabulary(f"{script_dir}/vocabulary")

    image_classifier = ImageClassifier(bow)

    # Load classifier
    image_classifier.load(classifier)
    print(image_classifier.predict(set, save=False))


def predict_new(classifier, dataset) -> list:
        """Evaluates a new set of images using the trained classifier.

        Args:
            trained classifier
            dataset: Paths to the test images.

        Returns:
            Classification results.

        """
        # Extract features
        test_desc = []

        for path in dataset:
            descriptors = classifier._extract_bow_features(path)

            if descriptors is not None:
                test_desc.extend(descriptors)

        # Predict categories
        predicted_labels = (classifier._classifier.predict(np.array(test_desc, np.float32))[1]).ravel().tolist()
        predicted_labels = [int(label) for label in predicted_labels]
        return predicted_labels


if __name__ == "__main__":
    # descriptors = extract_descriptors()
    # create_vocabulary(descriptors)

    # # Train the classifier
    # image_classifier, classifier = training_classificator()

    # #  Evaluate the model on the training set
    # print("CONJUNTO DE ENTRENAMIENTO:")
    # inference(classifier, training_set)

    # # Evaluate the model on the validation set
    # print("CONJUNTO DE VALIDACIÓN:")
    # inference(classifier, validation_set)

    # # Re-evaluate the model with all images (training + validation sets combined)
    # training_set = training_set + validation_set
    # descriptors = extract_descriptors()
    # create_vocabulary(descriptors)
    # image_classifier, classifier = training_classificator()

    image_classifier = initialise_classifier()
    # Load new data and predict using tñhe trained classifier
    new_data = Dataset.load(f"{script_dir}/new_image", "*.jpg")

    print(predict_new(image_classifier, new_data))
