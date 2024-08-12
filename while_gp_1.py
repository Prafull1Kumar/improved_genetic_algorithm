import random
from deap import base, creator, gp, tools, algorithms

# Define the primitive set
pset = gp.PrimitiveSet("MAIN", 0)
pset.addPrimitive(lambda x: x + 1, 1)
pset.addPrimitive(lambda x: x - 1, 1)
pset.addPrimitive(lambda x, y: x + y, 2)
pset.addPrimitive(lambda x, y: x - y, 2)
pset.addPrimitive(lambda x, y: x * y, 2)
pset.addPrimitive(lambda x, y: x // y if y != 0 else 1, 2)
pset.addPrimitive(lambda x, y: x % y if y != 0 else 1, 2)
pset.addPrimitive(lambda x: random.randint(0, x), 1)

# Add while loop structure to the primitive set
def if_then_else(input, output1, output2):
    if input:
        return output1
    else:
        return output2

pset.addPrimitive(if_then_else, 3)
pset.addPrimitive(lambda x, y: x < y, 2)
pset.addTerminal(True)
pset.addTerminal(False)

# Define the fitness function
def evaluate(individual):
    code = gp.compile(individual, pset)
    try:
        i = 0
        while code(i):
            i = code(i+1)
    except:
        pass
    fitness = i
    return fitness,

# Define the genetic programming algorithm
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)
toolbox = base.Toolbox()
toolbox.register("expr", gp.genFull, pset=pset, min_=1, max_=5)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("evaluate", evaluate)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("mutate", gp.mutNodeReplacement, pset=pset)

# Evolve the random program
population = toolbox.population(n=100)
for generation in range(20):
    offspring = algorithms.varAnd(population, toolbox, cxpb=0.5, mutpb=0.1)
    fits = toolbox.map(toolbox.evaluate, offspring)
    for ind, fit in zip(offspring, fits):
        ind.fitness.values = fit
    population = toolbox.select(offspring, k=len(population))

# Print the best individual
best_individual = tools.selBest(population, k=1)[0]
best_code = gp.compile(best_individual, pset)
i = 0
while best_code(i):
    i = best_code(i+1)
print("Final value of i: ", i)
