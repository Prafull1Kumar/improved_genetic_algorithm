import random
from math import sin, cos
from deap import creator, base, gp, tools, algorithms

# Define the fitness function
def eval_func(individual):
    # Evaluate the fitness of the individual
    x = random.uniform(-10, 10)
    result = eval(individual)
    return result,

# Define the primitive set of functions and terminals
functions = ['+', '-', '*', 'sin', 'cos']
terminals = ['x']
pset = {'functions': functions, 'terminals': terminals}

# Define the UNIF1SPACE operator
def unif1space(pset, individual):
    """
    The UNIF1SPACE operator mutates a gene in an individual by
    replacing it with a randomly chosen element from the primitive
    set with uniform probability.
    """
    index = random.randint(0, len(individual) - 1)
    if index >= len(pset['functions']):
        individual[index] = random.choice(pset['terminals'])
    else:
        individual[index] = random.choice(pset['functions'])
    return individual,

# Initialize the LGP algorithm
pop_size = 100
max_gen = 50
depth = 10
creator.create('FitnessMax', base.Fitness, weights=(1.0,))
creator.create('Individual', list, fitness=creator.FitnessMax)
toolbox = base.Toolbox()
toolbox.register('expr', gp.genHalfAndHalf, pset=pset, min_=1, max_=depth)
toolbox.register('individual', tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register('population', tools.initRepeat, list, toolbox.individual)
toolbox.register('compile', gp.compile, pset=pset)
toolbox.register('evaluate', eval_func)
toolbox.register('mate', gp.cxOnePoint)
toolbox.register('mutate', unif1space, pset=pset)
toolbox.register('select', tools.selTournament, tournsize=3)

# Set the random seed
random.seed(0)

# Run the LGP algorithm for a fixed number of generations
pop = toolbox.population(n=100)
hof = tools.HallOfFame(1)
stats = tools.Statistics(lambda ind: ind.fitness.values)
stats.register('avg', lambda x: sum(x) / len(x))
stats.register('min', lambda x: min(x))
stats.register('max', lambda x: max(x))
pop, log = algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.2, ngen=max_gen, stats=stats, halloffame=hof, verbose=True)

# Print the best individual
best_ind = hof[0]
print('Best individual:', best_ind)
