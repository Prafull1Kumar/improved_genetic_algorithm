import operator
import random
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
awareness_probability=0.5
flight_length=0.4

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
# toolbox.register("mutate", gp.mutNodeReplacement, pset=pset)
# toolbox.register("mutate", tools.mutPolynomialBounded, eta=1.0, low=-1.0, up=1.0, indpb=0.1)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr, pset=pset)

# Evolve the random expression
population = toolbox.population(n=100)
for generation in range(100):
    # offspring = algorithms.varAnd(population, toolbox, cxpb=0.5, mutpb=0.2)
    offspring = [toolbox.clone(ind) for ind in population]

    # Apply crossover and mutation on the offspring
    # 
    for i in range(1, len(offspring), 2):
        if random.random() < 0.5:
            offspring[i - 1], offspring[i] = toolbox.mate(offspring[i - 1],
                                                          offspring[i])
            del offspring[i - 1].fitness.values, offspring[i].fitness.values

    for i in range(len(offspring)):
        if random.random() < flight_length:
            offspring[i], = toolbox.mutate(offspring[i])
            del offspring[i].fitness.values

    
    fits = toolbox.map(toolbox.evaluate, offspring)
    for ind, fit in zip(offspring, fits):
        ind.fitness.values = fit

    for i in range(len(offspring)):
        if random.random() < awareness_probability:
            # if offspring[i].fitness.values[0]<toolbox.evaluate(population[i])[0]:
            population[i]=offspring[i]
            # if offspring[i].fitness.values[0]<toolbox.evaluate(population[i])[0]:
            #     population[i]=offspring[i]
    population = toolbox.select(population, k=len(population))        
    print(tools.selBest(population, k=1)[0].fitness.values[0])

# Print the best individual
best = tools.selBest(population, k=1)[0]
# print(gp.stringify(best))
print(best)
function = gp.compile(best, pset)
print(function(1, 2))
