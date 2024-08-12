import operator
import random
from deap import base, creator, tools

# Define the fitness function
def evaluate(individual):
    x, y = individual
    return (3*x + y*y),

# Define the problem and the population
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)
toolbox = base.Toolbox()
toolbox.register("attr_float", random.uniform, -10, 10)
toolbox.register("individual", tools.initRepeat, creator.Individual,
                 toolbox.attr_float, 2)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("evaluate", evaluate)

# Define the Crowding Distance function
def crowding_distance(population):
    # Sort individuals by fitness
    sorted_pop = sorted(population, key=lambda x: x.fitness.values[0])
    # Set the crowding distance for the first and last individuals to infinity
    sorted_pop[0].fitness.crowding_distance = float('inf')
    sorted_pop[-1].fitness.crowding_distance = float('inf')
    # Calculate the crowding distance for the rest of the individuals
    for i in range(1, len(population) - 1):
        sorted_pop[i].fitness.crowding_distance = (
            sorted_pop[i+1].fitness.values[0] -
            sorted_pop[i-1].fitness.values[0])

# Define the genetic operators
toolbox.register("mate", tools.cxUniform, indpb=0.5)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.2)
toolbox.register("select", tools.selTournament, tournsize=2)

# Define the Crow Search Algorithm
def crow_search_algorithm(population, toolbox, cxpb, mutpb, ngen, verbose=False):
    # Evaluate the initial population
    fitnesses = toolbox.map(toolbox.evaluate, population)
    for ind, fit in zip(population, fitnesses):
        ind.fitness.values = fit

    # Perform the evolution
    for g in range(ngen):
        # Create a new population by applying genetic operators to the current population
        offspring = toolbox.select(population, len(population))
        offspring = list(map(toolbox.clone, offspring))

        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < cxpb:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            if random.random() < mutpb:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        # Evaluate the new population
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # Merge the old and new populations
        population.extend(offspring)

        # Calculate the crowding distance of the individuals in the population
        crowding_distance(population)

        # Sort the population by fitness and crowding distance
        # population = sorted(population,
        #                     key=lambda x: (-x.fitness.values[0], x.fitness.c
                                           
        population = sorted(population,
                            key=lambda x: (-x.fitness.values[0], x.fitness.crowding_distance),
                            reverse=True)

        # Remove the least crowded individuals from the population
        population = population[:len(population)//2]

        if verbose:
            print("Generation:", g)
            print("Population:", population)  

        return population[0] 

# Set up the problem and run the algorithm
if __name__ == "__main__":
    # Set up the problem
    pop_size = 50
    cxpb = 0.7
    mutpb = 0.3
    ngen = 50
    random.seed(42)

    pop = toolbox.population(n=pop_size)

    # Run the algorithm
    best_ind = crow_search_algorithm(pop, toolbox, cxpb, mutpb, ngen, verbose=True)

    # Print the best individual
    print("Best individual:", best_ind)

    