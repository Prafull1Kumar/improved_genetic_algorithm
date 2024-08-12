import operator
import random
import numpy as np
from deap import gp, creator, base, tools,algorithms

# Define the primitive set of functions and terminals
pset = gp.PrimitiveSet("MAIN", arity=2)
pset.addPrimitive(operator.add, arity=2)
pset.addPrimitive(operator.mul, arity=2)
pset.addPrimitive(operator.neg, arity=1)
pset.addTerminal(3.0, name="three")
pset.addTerminal(4.0, name="four")
pset.renameArguments(ARG0="x")
pset.renameArguments(ARG1="y")
awareness_probability=0.25
flight_length=0.7

# Define the fitness function
def evaluate(individual):
    # print(individual)
    func = gp.compile(individual, pset)

    error = 0
    for x in range(-10, 11):
        for y in range(-10, 11):
            try:
                if func(x, y) is None:
                    raise TypeError
                error += abs(func(x, y) - (3*x*x + 4*y*y)) # Fitness function
            except:
                error += 100
    return error,

def manualMutUniform(individual, expr, pset):

    n=int(flight_length*len(individual))
    indexs=random.sample(range(0, len(individual)), n)
    # index = random.randrange(len(individual))
    # print(flight_length,len(individual),n)
    # print(indexs)
    for index in indexs:
        # print(index)
        if index<len(individual):
            slice_ = individual.searchSubtree(index)
            type_ = individual[index].ret
            # print(slice_,type_)
            individual[slice_] = expr(pset=pset, type_=type_)
    return individual,
# Define the genetic programming algorithm
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin,
               pset=pset)
toolbox = base.Toolbox()
toolbox.register("expr", gp.genFull, pset=pset, min_=2, max_=3)
toolbox.register("individual", tools.initIterate, creator.Individual,
                 toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("evaluate", evaluate)
toolbox.register("select", tools.selTournament, tournsize=4)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("mutate",manualMutUniform,expr=toolbox.expr, pset=pset)
# toolbox.register("mutate", gp.mutNodeReplacement, pset=pset)
# toolbox.register("mutate", tools.mutPolynomialBounded, eta=1.0, low=-1.0, up=1.0, indpb=0.1)
# toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr, pset=pset)

# Evolve the random expression
num_runs = 10
solutions_found=0
avg_generations=0
for run in range(num_runs):
    population = toolbox.population(n=100)
    for generation in range(100):
        # offspring = algorithms.varAnd(population, toolbox, cxpb=0.5, mutpb=0.2)
        offspring = [toolbox.clone(ind) for ind in population]

        for i in range(len(offspring)):
            if random.random() < awareness_probability:
                # if offspring[i].fitness.values[0]<toolbox.evaluate(population[i])[0]:
                # k = random.randint(0, i-1)
                # offspring[k], offspring[i] = toolbox.mate(offspring[k],offspring[i])
                # del offspring[k].fitness.values, offspring[i].fitness.values
                # population[i]=offspring[i]

                k=random.choice([k for k in range(0,9) if k!=i])
                if k<i:
                    offspring[k], offspring[i] = toolbox.mate(offspring[k],offspring[i])
                    del offspring[k].fitness.values, offspring[i].fitness.values
                else:
                    offspring[i], offspring[k] = toolbox.mate(offspring[i],offspring[k])
                    del offspring[k].fitness.values, offspring[i].fitness.values
                population[i]=offspring[i]
                # if offspring[i].fitness.values[0]<toolbox.evaluate(population[i])[0]:
                #     population[i]=offspring[i]
            else:
                if random.random() < 0.45:
                    # offspring[i]=gp.mutUniform(offspring[i],expr=toolbox.expr, pset=pset)

                    offspring[i], = toolbox.mutate(offspring[i])
                    del offspring[i].fitness.values
                population[i]=offspring[i]

        fits = toolbox.map(toolbox.evaluate, offspring)
        for ind, fit in zip(offspring, fits):
            ind.fitness.values = fit
        population = toolbox.select(population, k=len(population))        
        # print(tools.selBest(population, k=1)[0].fitness.values[0])
        best_fitness=tools.selBest(population, k=1)[0].fitness.values[0]
        if best_fitness == 0:
            solutions_found += 1
            avg_generations += generation + 1 # Record the generation in which the solution is found
            break

    # Print the best individual
    best = tools.selBest(population, k=1)[0]
    # print(gp.stringify(best))
    print(best)
    function = gp.compile(best, pset)
    print(function(1, 2))

print("Number of times solution found after 10 runs ",solutions_found)
avg_generations/=solutions_found
print("Avg generation: ",avg_generations)