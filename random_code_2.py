import operator
import random
import numpy as np
from deap import creator, base, gp, tools, algorithms

# Define the primitive set of functions and terminals
pset = gp.PrimitiveSet("MAIN", 2)
pset.addPrimitive(operator.add, 2)
pset.addPrimitive(operator.sub, 2)
pset.addPrimitive(operator.mul, 2)
pset.addPrimitive(operator.neg, 1)
pset.addTerminal(1)
pset.addTerminal(2)

# Define the fitness function
def eval_func(individual):
    func = gp.compile(individual, pset)
    # Evaluate the program on 2 random integers
    x1 = random.randint(1, 10)
    x2 = random.randint(1, 10)
    y = func(x1, x2)
    return y,

# Initialize the DEAP framework
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)
toolbox = base.Toolbox()
toolbox.register("expr", gp.genFull, pset=pset, min_=1, max_=2)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("evaluate", eval_func)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr, pset=pset)
toolbox.register("select", tools.selTournament, tournsize=3)

# Run the evolution
pop = toolbox.population(n=100)
hof = tools.HallOfFame(1)
stats = tools.Statistics(lambda ind: ind.fitness.values)
stats.register("avg", np.mean)
stats.register("min", np.min)
stats.register("max", np.max)
pop, logbook = algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.1, ngen=50, stats=stats, halloffame=hof, verbose=True)

# Print the best program found
best_program = gp.compile(hof[0], pset)
print(f"Best program: {hof[0]}")
print(f"Result of the program on 2 random integers: {best_program(random.randint(1, 10), random.randint(1, 10))}")
