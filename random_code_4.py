from deap import base, creator, tools, gp
import operator
import random
import ast

# Define the primitive set of functions and terminals
pset = gp.PrimitiveSet("MAIN", arity=0)
pset.addPrimitive(operator.add, arity=2)
pset.addPrimitive(operator.sub, arity=2)
pset.addPrimitive(operator.mul, arity=2)
pset.addPrimitive(operator.neg, arity=1)
pset.addTerminal(random.randint(0, 9), name="randint")
pset.addTerminal(1.0, name="one")
pset.addTerminal(2.0, name="two")

# Define the fitness function
def eval_func(individual):
    print(individual)
    # Evaluate the fitness of the individual by compiling and executing the code
    code = compile(individual, '<string>', 'exec')
    exec(code, globals())

    # Return the result of the executed code as fitness value
    return result,

# Define the LGP algorithm
class LGP:
    def __init__(self, toolbox, population_size, max_generations):
        self.toolbox = toolbox
        self.population_size = population_size
        self.max_generations = max_generations

    def run(self):
        # Create the initial population
        population = self.toolbox.population(n=self.population_size)

        for generation in range(self.max_generations):
            # Evaluate the fitness of the population
            # fitness_values = list(map(self.toolbox.evaluate, population))
            for individual in population:
                individual.fitness.values = gp.compile(individual, pset)

            # Select the parents for the next generation
            parents = self.toolbox.select(population, len(population))

            # Clone the parents to create the offspring
            offspring = list(map(self.toolbox.clone, parents))

            # Apply crossover and mutation on the offspring
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < 0.5:
                    self.toolbox.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values

            for mutant in offspring:
                if random.random() < 0.2:
                    self.toolbox.mutate(mutant)
                    del mutant.fitness.values

            # Evaluate the fitness of the offspring
            print(offspring)
            # fitness_values = list(map(self.toolbox.evaluate, offspring))
            for child in offspring:
                child.fitness.values = gp.compile(child, pset)

            # Replace the population with the offspring
            population[:] = offspring

        # Return the best individual in the final population
        return tools.selBest(population, k=1)

# Initialize the LGP algorithm
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)
toolbox = base.Toolbox()
toolbox.register("expr", gp.genFull, pset=pset, min_=1, max_=3)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", lambda ind: compile(ind, '<string>', 'exec'))
toolbox.register("evaluate", eval_func)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr, pset=pset)
toolbox.register("select", tools.selTournament, tournsize=3)

# Set the seed for random number generator
random.seed(0)

# Run the LGP algorithm for a fixed number of generations
population_size = 100
max_generations = 10
lgp = LGP(toolbox, population_size, max_generations)
best_individual = lgp.run()

# # Print the best individual and its fitness value
# print("Best individual:", best_individual[0])
# print("Fitness value:", best_individual[0].fitness.values[0])
