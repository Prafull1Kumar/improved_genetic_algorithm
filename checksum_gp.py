import operator
import random
import string

from deap import base, creator, gp, tools,algorithms

# Define the primitive set of functions and terminals
pset = gp.PrimitiveSet("MAIN", arity=1)
pset.addPrimitive(operator.add, arity=2)
pset.addPrimitive(operator.sub, arity=2)
pset.addPrimitive(operator.mul, arity=2)
pset.addPrimitive(operator.neg, arity=1)
pset.addTerminal(0, name="zero")
pset.addTerminal(1, name="one")
pset.addTerminal(2, name="two")
pset.addTerminal(3, name="three")
pset.addTerminal(4, name="four")
pset.addTerminal(5, name="five")
pset.addTerminal(6, name="six")
pset.addTerminal(7, name="seven")
pset.addTerminal(8, name="eight")
pset.addTerminal(9, name="nine")
pset.addTerminal(string.ascii_lowercase, name="letters")

# Define the fitness function
def eval_func(individual):
    # Compile the program
    program = gp.compile(individual, pset)

    # Define the input string
    input_string = "example"

    # Calculate the checksum of the input string
    checksum = sum([ord(c) for c in input_string])

    # Evaluate the program and compare the output with the expected checksum
    try:
        output = program(input_string)
        error = abs(output - checksum)
    except:
        error = float('inf')

    # Return the fitness as a tuple
    return error,

# Define the GP algorithm
def run_gp():
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)

    toolbox = base.Toolbox()
    toolbox.register("expr", gp.genFull, pset=pset, min_=1, max_=3)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("compile", gp.compile, pset=pset)
    toolbox.register("evaluate", eval_func)
    toolbox.register("mate", gp.cxOnePoint)
    toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr, pset=pset)
    toolbox.register("select", tools.selTournament, tournsize=3)

    population_size = 100
    max_generations = 50

    population = toolbox.population(n=population_size)

    for generation in range(max_generations):
        offspring = algorithms.varAnd(population, toolbox, cxpb=0.5, mutpb=0.2)
        fits = toolbox.map(toolbox.evaluate, offspring)
        for fit, ind in zip(fits, offspring):
            ind.fitness.values = fit
        population = toolbox.select(offspring, k=len(population))

    best_individual = tools.selBest(population, k=1)[0]
    print(best_individual)
    best_program = gp.compile(best_individual, pset)
    return best_program

# Run the GP algorithm and print the best program
best_program = run_gp()
# tree1=gp.PrimitiveTree(best_program)
# print(tree1)
print("Best program:", best_program)
