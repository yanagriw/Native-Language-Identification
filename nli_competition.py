#!/usr/bin/env python3

import argparse
import lzma
import pickle
import os
from typing import Optional

import numpy as np
import numpy.typing as npt
import sklearn.model_selection
import sklearn.pipeline
import sklearn.feature_extraction
import sklearn.neural_network

parser = argparse.ArgumentParser()
parser.add_argument("--predict", default=None, type=str, help="Path to the dataset to predict")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
parser.add_argument("--model_path", default="nli_competition.model", type=str, help="Model path")


class Dataset:
    CLASSES = ["ARA", "DEU", "FRA", "HIN", "ITA", "JPN", "KOR", "SPA", "TEL", "TUR", "ZHO"]

    def __init__(self, name="nli_dataset.train.txt"):
        if not os.path.exists(name):
            raise RuntimeError("The {} was not found, please download it from ReCodEx".format(name))

        # Load the dataset and split it into `data` and `target`.
        self.data, self.prompts, self.levels, self.target = [], [], [], []
        with open(name, "r", encoding="utf-8") as dataset_file:
            for line in dataset_file:
                target, prompt, level, text = line.rstrip("\n").split("\t")
                self.data.append(text)
                self.prompts.append(prompt)
                self.levels.append(level)
                self.target.append(-1 if not target else self.CLASSES.index(target))
        self.target = np.array(self.target, np.int32)


def main(args: argparse.Namespace) -> Optional[npt.ArrayLike]:
    if args.predict is None:
        # We are training a model.
        np.random.seed(args.seed)
        train = Dataset()

        # Train a model on the given dataset and store it in `model`.
        for i in range(len(train.data)):
            train.data[i] = train.data[i] + ' ' + train.prompts[i]

        # Train a model on the given dataset and store it in model.
        model = sklearn.pipeline.Pipeline(
            [('preproc_char', sklearn.feature_extraction.text.TfidfVectorizer(analyzer='word', ngram_range=(1, 2),
                                                                              sublinear_tf=True, max_features=60000)),
             ('model', sklearn.neural_network.MLPClassifier(hidden_layer_sizes=150, verbose=True))])

        model.fit(train.data, train.target)
        model['model']._optimizer = None
        for i in range(len(model['model'].coefs_)): model['model'].coefs_[i] = model['model'].coefs_[i].astype(
            np.float16)
        for i in range(len(model['model'].intercepts_)): model['model'].intercepts_[i] = model['model'].intercepts_[i].astype(
            np.float16)

        # Serialize the model.
        with lzma.open(args.model_path, "wb") as model_file:
            pickle.dump(model, model_file)

    else:
        # Use the model and return test set predictions.
        test = Dataset(args.predict)
        for i in range(len(test.data)):
            test.data[i] = test.data[i] + ' ' + test.prompts[i]

        with lzma.open(args.model_path, "rb") as model_file:
            model = pickle.load(model_file)

        # Generate `predictions` with the test set prediction
        predictions = model.predict(test.data)

        return predictions


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
