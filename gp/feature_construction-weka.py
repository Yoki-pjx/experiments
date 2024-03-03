import gc
import operator
import random
import numpy as np
import pandas as pd
from functools import partial

from deap import base, creator, tools, gp, algorithms

import weka.core.jvm as jvm
import WeakClassifier

# 遗传算法参数
random.seed(166)
pop_size = 500
crossover_pb = 0.9
mutation_pb = 0.1
frac_elitist = 0.1
n_feature = 5
ngen = 100
max_depth = 17

# load dataset
data = pd.read_csv("../Data/total_data.csv")  
x_values = data.iloc[:, :20].values  
y_values = data.iloc[:, 20].values

# Create a class of fitness and a class of individuals, with individuals consisting of multiple trees
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

jvm.start(system_cp=True, packages=True)

# Baseline calculation
scores_base = WeakClassifier.REPTree(x_values, y_values)

# The goal of solving the problem is to maximize classification accuracy
def evalFeatureEngineering(individuals):
    # Creating new features
    new_features = []
    for ind_num, ind in enumerate(individuals):
        func = gp.compile(expr=ind, pset=pset)
        new_features.append([func(*record) for record in x_values])
    
    # Transpose New Feature Array
    new_features = np.transpose(np.array(new_features))
    
    # Decision Tree Classifier
    evl_scores = WeakClassifier.REPTree(new_features, y_values)
    print(evl_scores)

    del new_features
    gc.collect()
   
    # Returns accuracy
    return evl_scores


# Define new functions
# def protectedDiv(left, right):
#     if right == 0:
#         return 1
#     return left / right

def less(left, right):
    if left < right:
        return 1
    else:
        return 0
    
def great(left, right):
    if left > right:
        return 1
    else:
        return 0

# def equal(left, right):
#     if left == right:
#         return 1
#     else:
#         return 0        

# 创建GP框架的基本组件
pset = gp.PrimitiveSet("MAIN", x_values.shape[1])
pset.addPrimitive(operator.add, 2)
pset.addPrimitive(operator.sub, 2)
# pset.addPrimitive(operator.mul, 2)
# pset.addPrimitive(operator.neg, 1)
# pset.addPrimitive(operator.abs, 1)
# pset.addPrimitive(protectedDiv, 2)
pset.addPrimitive(less, 2)
pset.addPrimitive(great, 2)
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
        gp.cxOnePoint(tree1, tree2)
        if tree1.height > max_depth or tree2.height > max_depth:
            # If the depth of any tree exceeds the limit, the original individual is restored
            return original_ind1, original_ind2
    return ind1, ind2

def mutUniformListOfTrees(individual, expr, pset, max_depth=max_depth):
    original_individual = toolbox.clone(individual)

    for tree in individual:
        gp.mutUniform(tree, expr=expr, pset=pset)
        if tree.height > max_depth:
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

# Genetic Programming Algorithms
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

jvm.stop()

print(f'Weka.REPTree p {pop_size}, ngen {ngen}, crossover {crossover_pb}, mutation {mutation_pb}, Elitist {frac_elitist}, feature {n_feature}')

print('Log:')
print(log)

# print('Hof:')
# for ind in hof:
#     print(str(ind))

print('Baseline Acc:', scores_base)

# Best individual
best_ind=hof[0]
print('Best individual is:', [str(tree) for tree in best_ind])
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

sixth_ind=hof[5]
print('6th individual is:', [str(tree) for tree in sixth_ind])
print('With fitness:', sixth_ind.fitness.values)

seventh_ind=hof[6]
print('7th best individual is:', [str(tree) for tree in seventh_ind])
print('With fitness:', seventh_ind.fitness.values)

eighth_ind=hof[7]
print('8th best individual is:', [str(tree) for tree in eighth_ind])
print('With fitness:', eighth_ind.fitness.values)

ninth_ind=hof[8]
print('9th best individual is:', [str(tree) for tree in ninth_ind])
print('With fitness:', ninth_ind.fitness.values)

tenth_ind=hof[9]
print('10th best individual is:', [str(tree) for tree in tenth_ind])
print('With fitness:', tenth_ind.fitness.values)