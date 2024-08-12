import random
# Define the input and output types
import operator
# from gplearn import algorithms
# from deap import gp
from deap import creator, base,algorithms,gp,tools
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
    # Evaluate the fitness of the individual
    return individual[0] * individual[1],

# Initialize the LGP algorithm
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

lgp = algorithms.LGP(toolbox, population_size=100, max_generations=50)


inputs = [(int, 'a'), (int, 'b')]
output = (int,)

# Define the function set
functions = {
    'add': lambda a, b: a + b,
    'sub': lambda a, b: a - b,
    'mul': lambda a, b: a * b,
    'div': lambda a, b: a // b if b != 0 else 0,
    'mod': lambda a, b: a % b if b != 0 else 0,
    'and': lambda a, b: a and b,
    'or': lambda a, b: a or b,
    'not': lambda a: not a,
    'xor': lambda a, b: a != b,
    'eq': lambda a, b: a == b,
    'ne': lambda a, b: a != b,
    'lt': lambda a, b: a < b,
    'le': lambda a, b: a <= b,
    'gt': lambda a, b: a > b,
    'ge': lambda a, b: a >= b,
}

# Define the terminals
terminals = {
    'a': 'a',
    'b': 'b',
    'const': lambda: random.randint(1, 10),
}

# Define the maximum depth of the trees
max_depth = 5

# Define the population size
population_size = 100

# Define the number of generations
num_generations = 100

# Define the fitness function
def fitness_function(program):
    total_error = 0
    for i in range(10):
        a = random.randint(1, 10)
        b = random.randint(1, 10)
        expected_output = a * b
        actual_output = program(a=a, b=b)
        total_error += abs(expected_output - actual_output)
    return total_error

# Define the selection function
def selection(population):
    return random.choice(population)

# Define the crossover function
def crossover(parent1, parent2):
    return lambda **kwargs: (parent1(**kwargs), parent2(**kwargs))

# Define the mutation function
def mutation(program):
    return lambda **kwargs: random.choice(list(terminals.values()))() if random.random() < 0.1 else program(**kwargs)

# Initialize the population
population = []
for i in range(population_size):
    program = lgp.generate(random.randint(1, max_depth), functions, terminals)
    fitness = fitness_function(program)
    population.append((program, fitness))

# Evolve the population
for generation in range(num_generations):
    # Select two parents
    parent1 = selection(population)
    parent2 = selection(population)

    # Crossover the parents
    child = crossover(parent1[0], parent2[0])

    # Mutate the child
    child = mutation(child)

    # Evaluate the child's fitness
    fitness = fitness_function(child)

    # Replace a random member of the population with the child
    population[random.randint(0, population_size - 1)] = (child, fitness)

# Select the best program
best_program = min(population, key=lambda x: x[1])[0]

# Print the program
print(best_program(a=5, b=6))
