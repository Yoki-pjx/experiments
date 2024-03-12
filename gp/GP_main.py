import os, tempfile, json
import subprocess

import random
import numpy as np
import pandas as pd

import operator
from deap import base, creator, tools, gp, algorithms

# Parameters
random.seed(166)
pop_size = 500
crossover_pb = 0.9
mutation_pb = 0.1
frac_elitist = 0.1
n_feature = 5
ngen = 100
max_depth = 17

# Load dataset
data = pd.read_csv("../Data/total_data.csv")  
# data = pd.read_csv("./10k.csv")  

x_values = data.iloc[:, :20].values  
y_values = data.iloc[:, 20].values

print('Dataset loaded...')

# Create a class of fitness and a class of individuals, with individuals consisting of multiple trees
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

# Call subprocess
def call_REPTree(x_values, y_values):
    try:
        # print("Preparing data for subprocess...")
        x_file = tempfile.NamedTemporaryFile(delete=False, mode='w', suffix='.json')
        y_file = tempfile.NamedTemporaryFile(delete=False, mode='w', suffix='.json')

        # data to temp files
        json.dump(x_values.tolist() if isinstance(x_values, np.ndarray) else x_values, x_file)
        json.dump(y_values.tolist() if isinstance(y_values, np.ndarray) else y_values, y_file)
        x_file_path = x_file.name
        y_file_path = y_file.name
        x_file.close()
        y_file.close()

        # creat temp file for score
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file_score = temp_file.name

        # print("Starting subprocess...")
        result = subprocess.Popen(['python', 
                'REPTree.py', x_file_path, y_file_path, temp_file_score], 
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE, 
                text=True)
        
        # stand output
        # while True:
        #     output = result.stdout.readline()
        #     if output == '' and result.poll() is not None:
        #         break
        #     if output:
        #         print(output.strip())

        # # stand error
        # stderr = result.stderr.read()
        # if stderr:
        #     print(stderr.strip())

        # wait for subprocess
        exit_code = result.wait()
        # print(f"Subprocess finished with exit code {exit_code}")
        # print("Subprocess finished.")

        # reutn score
        if result.returncode == 0:
            with open(temp_file_score, 'r') as file:
                score = float(file.read().strip()) 
            # print("Received score from subprocess:", score)
            return score
        else:
            raise Exception(f"Subprocess failed with: {result.stderr}")
    finally:
        # remove temp files
        os.remove(x_file_path)
        os.remove(y_file_path)
        os.remove(temp_file_score)

# The goal of solving the problem is to maximize classification accuracy
# def evalFeatureEngineering(individuals):
#     # Creating new features
#     new_features = []
#     for ind_num, ind in enumerate(individuals):
#         func = gp.compile(expr=ind, pset=pset)
#         new_features.append([func(*record) for record in x_values])
    
#     # Transpose New Feature Array
#     new_features = np.transpose(np.array(new_features))
    
#     # Decision Tree Classifier
#     evl_scores = WeakClassifier.REPTree(new_features, y_values)
#     print(evl_scores)

#     del new_features
#     gc.collect()
   
#     # Returns accuracy
#     return [evl_scores]

def evalFeatureEngineering(individuals):
    new_features = []

    # Create new features
    for ind_num, ind in enumerate(individuals):
        func = gp.compile(expr=ind, pset=pset)
        new_features.append([func(*record) for record in x_values])
    
    # Transpose New Feature Array
    new_features = np.transpose(np.array(new_features))
    
    # Decision Tree Classifier
    evl_scores = call_REPTree(new_features, y_values)

    print(evl_scores)
    
    # Returns accuracy
    return [evl_scores]

# Baseline calculation
print('Modeling the baseline...')
scores_base = call_REPTree(x_values, y_values)
# print(scores_base)

# Define primitive
# def protectedDiv(left, right):
#     if right == 0:
#         return 1
#     return left / right

# def less(left, right):
#     if left < right:
#         return 1
#     else:
#         return 0
    
# def great(left, right):
#     if left > right:
#         return 1
#     else:
#         return 0

# def equal(left, right):
#     if left == right:
#         return 1
#     else:
#         return 0        

# GP setting
pset = gp.PrimitiveSet("MAIN", x_values.shape[1])
pset.addPrimitive(operator.add, 2)
pset.addPrimitive(operator.sub, 2)
# pset.addPrimitive(operator.mul, 2)
# pset.addPrimitive(operator.neg, 1)
# pset.addPrimitive(operator.abs, 1)
# pset.addPrimitive(protectedDiv, 2)
pset.addPrimitive(max, 2)
pset.addPrimitive(min, 2)
# pset.addPrimitive(equal, 2)

def low_precision_random():
    return round(random.uniform(-1, 1), 6)

pset.addEphemeralConstant("uni101", low_precision_random)
# pset.addEphemeralConstant("rand101", lambda: random.random() * 2 - 1)

def selElitistAndTournament(individuals, k, frac_elitist, tournsize):
    return tools.selBest(individuals, int(k*frac_elitist)) + tools.selTournament(individuals, int(k*(1-frac_elitist)), tournsize=tournsize)

toolbox = base.Toolbox()
toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=5)

# Initialize each individual to a list containing multiple trees
def initIndividual(container, func, size):
    return container(gp.PrimitiveTree(func()) for _ in range(size))

# The crossover and mutation operators need to be able to handle the list structure of individuals
def cxOnePointListOfTrees(ind1, ind2, max_depth=max_depth):
    # Cloning of the original individual
    original_ind1 = toolbox.clone(ind1)
    original_ind2 = toolbox.clone(ind2)

    for tree1, tree2 in zip(ind1, ind2):
        tree1, tree2 = gp.cxOnePoint(tree1, tree2)
        if tree1.height > max_depth or tree2.height > max_depth:
            # If the depth of any tree exceeds the limit, the original individual is restored
            return original_ind1, original_ind2          
        
    return ind1, ind2

def mutUniformListOfTrees(individual, expr, pset, max_depth=max_depth):
    
    original_individual = toolbox.clone(individual)

    for tree in individual:
        mutated_tree, = gp.mutUniform(tree, expr=expr, pset=pset)
        if mutated_tree.height > max_depth:
            # If the depth of the tree exceeds the limit, the original individual is restored
            return original_individual,
    
    return individual,

toolbox.register("individual", initIndividual, creator.Individual, toolbox.expr, size=n_feature)  # Create n features
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("evaluate", evalFeatureEngineering)
# toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("select", selElitistAndTournament, frac_elitist=frac_elitist , tournsize=3)
toolbox.register("mate", cxOnePointListOfTrees)
toolbox.register("mutate", mutUniformListOfTrees, expr=toolbox.expr, pset=pset)
toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
toolbox.register("compile", gp.compile, pset=pset)


# Genetic Programming algorithms
population = toolbox.population(n=pop_size)
hof = tools.HallOfFame(10)

# Log
stats = tools.Statistics(lambda ind: ind.fitness.values)
stats.register("avg", np.mean)
stats.register("std", np.std)
stats.register("min", np.min)
stats.register("max", np.max)

pop, log = algorithms.eaSimple(population, toolbox,
                                cxpb=crossover_pb, mutpb=mutation_pb, ngen=ngen,
                                stats=stats, halloffame=hof, verbose=True)


print('--' * 40)
print(f'Weka.REPTree p {pop_size}, ngen {ngen}, crossover {crossover_pb}, mutation {mutation_pb}, Elitist {frac_elitist}, feature {n_feature}')
print('--' * 40)
print('Baseline Acc:', scores_base)
print('--' * 40)

print('Log:')
print(log)
print('--' * 40)

# Best individual
best_ind=hof[0]
print('\nBest individual is:', [str(tree) for tree in best_ind])
print('With fitness:', best_ind.fitness.values)

sec_ind=hof[1]
print('2rd best individual is:', [str(tree) for tree in sec_ind])
print('With fitness:', sec_ind.fitness.values)

third_ind=hof[2]
print('3rd best individual is:', [str(tree) for tree in third_ind])
print('With fitness:', third_ind.fitness.values)

forth_ind=hof[3]
print('4th best individual is:', [str(tree) for tree in forth_ind])
print('With fitness:', forth_ind.fitness.values)

fifth_ind=hof[4]
print('5th best individual is:', [str(tree) for tree in fifth_ind])
print('With fitness:', fifth_ind.fitness.values)