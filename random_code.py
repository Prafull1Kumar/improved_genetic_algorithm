import random
import operator
import ast
from deap import gp, algorithms, base, creator, tools

# Define the primitive set of functions and terminals
pset = gp.PrimitiveSet("MAIN", arity=0)
pset.addPrimitive(operator.add, arity=2)
pset.addPrimitive(operator.sub, arity=2)
pset.addPrimitive(operator.mul, arity=2)
pset.addPrimitive(operator.neg, arity=1)
pset.addPrimitive(max, arity=2)
pset.addPrimitive(min, arity=2)
pset.addPrimitive(sum, arity=1)
pset.addPrimitive(len, arity=1)
pset.addTerminal(random.randint(0, 9), name="randint")
pset.addTerminal("hello", name="hello")
pset.addTerminal("world", name="world")

# Define the fitness function
def eval_func(individual):
    # Evaluate the fitness of the individual by compiling and executing the code
    code = compile(individual, '<string>', 'exec')
    exec(code, globals())

    # Return the result of the executed code as fitness value
    return result,

# Initialize the LGP algorithm
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)
toolbox = base.Toolbox()
toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=5)
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
lgp = algorithms.LGP(toolbox, population_size=population_size, max_generations=max_generations)

# # Retrieve the best individual and compile the code
# best_individual = lgp.run()[0]
# code = compile(best_individual, '<string>', 'exec')

# # Print the generated code
# print(ast.dump(ast.parse(best_individual)))
