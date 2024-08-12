import operator
import random
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
# Create a tree of primitives
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin,
               pset=pset)
# Define the mutation operator
toolbox = base.Toolbox()
toolbox.register("mutate", gp.mutNodeReplacement, pset=pset)

# Create an initial individual
ind = gp.PrimitiveTree.from_string("(add (mul x y) (neg three))", pset)
print("Original individual:", ind)

# Mutate the individual to create a new individual
new_ind= toolbox.mutate(ind)
# if success:
print("Mutated individual:", new_ind)
# else:
#     print("Mutation failed, no new individual created")