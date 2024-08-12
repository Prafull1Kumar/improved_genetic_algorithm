import random
import operator
import deap.gp as gp
from deap import creator, base, tools, algorithms

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
    # Evaluate the fitness of the individual by compiling and executing the code
    print(2)
    code = compile(individual, '<string>', 'exec')
    exec(code, globals())

    # Return the result of the executed code as fitness value
    return result,

# Define the GP algorithm
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
print(1)
# Generate a random code
ind = toolbox.individual()
print(ind)
code = gp.compile(ind, pset)

# Print the generated code
print(code)
