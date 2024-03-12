import sys
import json
import numpy as np

import weka.core.jvm as jvm
from weka.core.dataset import create_instances_from_matrices
from weka.filters import Filter
from weka.classifiers import Classifier
from weka.classifiers import Evaluation
from weka.core.classes import Random


def REPTree(x_values, y_values, options=["-L", "10"]):
    # print("Creating dataset for Weka...")
    dataset = create_instances_from_matrices(x_values, y_values, name="features")
    dataset.class_is_last()

    # print("Filteringing dataset for Weka...")
    flted = Filter(classname="weka.filters.unsupervised.attribute.NumericToNominal", options=["-R", "last"])
    flted.inputformat(dataset)
    filtered = flted.filter(dataset)
    
    # Creating classifier
    cls = Classifier(classname="weka.classifiers.trees.REPTree", options=options)
    # cls.build_classifier(filtered)

    # print("Evaluating dataset for Weka...")
    evl = Evaluation(filtered)
    evl.discard_predictions = True
    evl.crossvalidate_model(cls, filtered, 10, Random(1))
    score = evl.correct / evl.num_instances
    
    return score

if __name__ == "__main__":
    # print("Loading data for subprocess...")
    x_file_path = sys.argv[1]
    y_file_path = sys.argv[2]
    temp_file_score = sys.argv[3]

    # print("Opening data for subprocess...")    
    with open(x_file_path, 'r') as x_file, open(y_file_path, 'r') as y_file:
        x_values_list = json.load(x_file)
        y_values_list = json.load(y_file)

    # Convert list to ndarray
    x_values = np.array(x_values_list)
    y_values = np.array(y_values_list)

    # print("Starting JVM...")
    jvm.start(system_cp=True, packages=True)

    score = REPTree(x_values, y_values)
    # print(score)
    
    # Save score to tempfile
    with open(temp_file_score, 'w') as temp_file:
        temp_file.write(str(score))

    jvm.stop()