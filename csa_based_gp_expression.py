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
# toolbox.register("mate", tools.cxTwoPoint)
# toolbox.register("mate", tools.cxBlend, alpha=0.2)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr, pset=pset)
# toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.2)
toolbox.register("select", tools.selTournament, tournsize=3)

# Define the Crow Search Algorithm
def crow_search_algorithm(population, evaluate, iterations, beta):
    for i in range(iterations):
        # Calculate the fitness of each individual
        fitnesses = [evaluate(individual) for individual in population]

        # Sort the population by fitness
        # print("Fitnesses",fitnesses)
        # print("Initial Population",population)
        population = [individual for _, individual in zip(fitnesses, population)]
        population=sorted(population, key=lambda x: x.fitness.values[0])
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
            # print(population[j].fitness.values[0])
            if random.random() < beta:
                selected_individuals.append(population[j])
            else:
                k = random.randint(0, len(population)-1)
                if distances[j] >= distances[k]:
                    selected_individuals.append(population[j])
                else:
                    selected_individuals.append(population[k])

        offspring = []

        offspring = algorithms.varAnd(population, toolbox, cxpb=0.5, mutpb=0.2)
        population = offspring
        
        # Create the new population by crossover and mutation
        # offspring = []
        # for j in range(0, len(selected_individuals)-1, 2):
        #     child1, child2 = tools.cxUniform(selected_individuals[j], selected_individuals[j+1],0.5)
        #     del child1.fitness.values
        #     del child2.fitness.values
        #     offspring.append(child1)
        #     offspring.append(child2)

        # for j in range(0, len(selected_individuals)-1, 2):
        #     child1, child2 = toolbox.mate(selected_individuals[j], selected_individuals[j+1])
        #     del child1.fitness.values
        #     del child2.fitness.values
        #     offspring.append(child1)
        #     offspring.append(child2)
            

        # for child1, child2 in zip(offspring[::2], offspring[1::2]):
        #     if random.random() < 0.5:
        #         child1,child2=toolbox.mate(child1, child2)
        #         offspring.append(child1)
        #         offspring.append(child2)
        #         del child1.fitness.values
        #         del child2.fitness.values
        # for j in range(len(offspring)):
        #     if random.random() < 0.1:
        #         offspring[j] = tools.mutUniform(offspring[j], expr=pset)
        # for j in range(len(population)):
        #     if random.random() < 0.1:
        #         population[j] = toolbox.mutate(population[j])
                # del population[j].fitness.values
        # for mutant in offspring:
        #     if random.random() < 0.1:
        #         before=mutant
        #         chnaged=toolbox.mutate(mutant)
        #         after=mutant
        #         # print(chnaged)
        #         if chnaged!=before:
        #             print("changed")
        #         if after==before:
        #             print("same")    
        #         del mutant.fitness.values
        # for j in range(len(offspring)):
        #     if random.random() < 0.1:
        #         # offspring[j] = toolbox.mutate(offspring[j], expr=pset)
        #         toolbox.mutate(offspring[j])

        # population = offspring

        # offspring = []
        # for j in range(0, len(selected_individuals)-1, 2):
        #     offspring.append(selected_individuals[j])
        #     child1, child2 = tools.cxPartialyMatched(selected_individuals[j], selected_individuals[j+1],0.5)
        #     offspring.append(child1)
        #     offspring.append(child2)

        # for j in range(len(offspring)):
        #     if random.random() < 0.1:
        #         offspring[j] = tools.mutUniformInt(offspring[j], expr=pset)

        # population = offspring

        


    return population


# Define the main function for the genetic programming algorithm
def main():
    pop = toolbox.population(n=20)
    beta = 0.2
    CXPB, MUTPB, NGEN = 0.5, 0.2, 100

    # Evaluate the entire population
    fitnesses = list(map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit

    # Begin the evolution
    for g in range(NGEN):
        # Select the next generation individuals
        offspring = crow_search_algorithm(pop, toolbox.evaluate, 1, beta)

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # Replace the population with the offspring
        pop[:] = offspring

        # Print the statistics for each generation
        fits = [ind.fitness.values[0] for ind in pop]
        length = len(pop)
        mean = sum(fits) / length
        # sum2 = sum(x*x for x in fits)
        # std = abs(sum2 / length - mean**2)**0.5
        print(f"Generation {g}: Min {min(fits)}, Max {max(fits)}, Avg {mean}")

    best_ind = tools.selBest(pop, 1)[0]
    print("Best individual:", best_ind)
    print("Fitness:", best_ind.fitness.values[0])
    return best_ind

if __name__ == "__main__":
    main()
