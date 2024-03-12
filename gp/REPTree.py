import gc
import weka.core.jvm as jvm
from weka.filters import Filter
from weka.classifiers import Classifier
from weka.core.dataset import create_instances_from_matrices
from weka.classifiers import Evaluation
from weka.core.classes import Random


def J48(x_values, y_values):
    # Load data
    # data = pd.read_csv("F:/programming/GP/10kfull.csv")  
    # x_values = data.iloc[:, :20].values  
    # y_values = data.iloc[:, 20].values


    dataset = create_instances_from_matrices(x_values, y_values, name="new_feature")
    dataset.class_is_last()
    
    flted = Filter(classname="weka.filters.unsupervised.attribute.NumericToNominal", options=["-R", "last"])
    flted.inputformat(dataset) 
    filtered = flted.filter(dataset)

    cls = Classifier(classname="weka.classifiers.trees.J48", options=["-C", "0.25"])
    cls.build_classifier(filtered)

    
    evl = Evaluation(filtered)
    evl.crossvalidate_model(cls, filtered, 10, Random(1))
    score = evl.correct / evl.num_instances
    # print(evl.correct / evl.num_instances)
    # print(score)

    del dataset, filtered
    gc.collect()

    # print(evl.percent_correct)
    # print(evl.summary())
    # print(evl.class_details())
    return score,
    

# def REPTree(x_values, y_values):
#     # Load data
#     # data = pd.read_csv("F:/programming/GP/10kfull.csv")  
#     # x_values = data.iloc[:, :20].values  
#     # y_values = data.iloc[:, 20].values

#     dataset = create_instances_from_matrices(x_values, y_values, name="new_feature")
#     dataset.class_is_last()
    
#     flted = Filter(classname="weka.filters.unsupervised.attribute.NumericToNominal", options=["-R", "last"])
#     flted.inputformat(dataset) 
#     filtered = flted.filter(dataset)

#     cls = Classifier(classname="weka.classifiers.trees.REPTree", options=["-L", "10"])
#     cls.build_classifier(filtered)
  
#     evl = Evaluation(filtered)
#     evl.crossvalidate_model(cls, filtered, 10, Random(1))
#     score = evl.correct / evl.num_instances
#     # print(evl.correct / evl.num_instances)
#     # print(score)
#     del dataset, filtered
#     gc.collect()

#     # print(evl.percent_correct)
#     # print(evl.summary())
#     # print(evl.class_details())
#     return score


def REPTree(x_values, y_values, options=["-L", "10"]):
    try:
        dataset = create_instances_from_matrices(x_values, y_values, name="new_feature")
        dataset.class_is_last()
        
        flted = Filter(classname="weka.filters.unsupervised.attribute.NumericToNominal", options=["-R", "last"])
        flted.inputformat(dataset)
        filtered = flted.filter(dataset)

        cls = Classifier(classname="weka.classifiers.trees.REPTree", options=options)
        cls.build_classifier(filtered)
        
        evl = Evaluation(filtered)
        evl.crossvalidate_model(cls, filtered, 10, Random(1))
        score = evl.correct / evl.num_instances
    finally:
        del flted, filtered
        gc.collect()

    return score