import operator
import random
import math
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

# Define the Crow Search Algorithm
def crow_search_algorithm(population, evaluate, iterations, beta):
    for i in range(iterations):
        # Calculate the fitness of each individual
        fitnesses = [evaluate(individual) for individual in population]

        # Sort the population by fitness
        population = [individual for _, individual in sorted(zip(fitnesses, population))]

        # Calculate the crowding distance of each individual
        distances = [0] * len(population)
        for j in range(len(population[0].fitness.values)):
            sorted_population = sorted(population, key=lambda x: x.fitness.values[j])
            distances[0] = math.inf
            distances[-1] = math.inf
            for k in range(1, len(population)-1):
                distances[k] += (sorted_population[k+1].fitness.values[j] - sorted_population[k-1].fitness.values[j])
        
        # Select the best individuals based on the crowding distance
        selected_individuals = []
        for j in range(len(population)):
            if random.random() < beta:
                selected_individuals.append(population[j])
            else:
                k = random.randint(0, len(population)-1)
                if distances[j] >= distances[k]:
                    selected_individuals.append(population[j])
                else:
                    selected_individuals.append(population[k])

        # Create the new population by crossover and mutation
        offspring = []
        for j in range(0, len(selected_individuals)-1, 2):
            child1, child2 = tools.cxOnePoint(selected_individuals[j], selected_individuals[j+1])
            offspring.append(child1)
            offspring.append(child2)

        for j in range(len(offspring)):
            if random.random() < 0.1:
                offspring[j] = tools.mutUniform(offspring[j], expr=pset)

        population = offspring

    return population

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
# Define the operators
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("mutate", gp.mutUniform, expr=pset, pset=pset)
toolbox.register("select", tools.selTournament, tournsize=3)

# Define the main function for the genetic programming algorithm
def main():
    random.seed(42)

    # Create an initial population
    pop = toolbox.population(n=100)

    # Run the genetic programming algorithm
    algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.2, ngen=50, verbose=False)

    # Evaluate the best individual
    best_ind = tools.selBest(pop, k=1)[0]
    func = gp.compile(best_ind, pset)

    # Print the best individual and its fitness value
    print("Best individual:", best_ind)
    print("Fitness value:", evaluate(best_ind)[0])

    # Print the values of the function for some inputs
    print("Function values:")
    for x in range(-10, 11):
        for y in range(-10, 11):
            print("f({}, {}) = {}".format(x, y, func(x, y)))

if __name__ == "__main__":
    main()
