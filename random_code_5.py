import operator
import random
import json
from deap import gp,creator,base,tools

# Define the primitive set of functions and terminals
pset = gp.PrimitiveSet("MAIN", arity=2)
pset.addPrimitive(operator.add, arity=2)
pset.addPrimitive(operator.sub, arity=2)
pset.addPrimitive(operator.mul, arity=2)
pset.addPrimitive(operator.neg, arity=1)
# pset.addTerminal(random.randint(0, 9), name="randint")
pset.addTerminal(1.0, name="one")
pset.addTerminal(2.0, name="two")
pset.renameArguments(ARG0="x")
pset.renameArguments(ARG1="y")

# def if_then_else(input, output1, output2):
#     if input:
#         return output1
#     else:
#         return output2

# pset.addPrimitive(if_then_else, 3)
# pset.addPrimitive(lambda x, y: x < y, 2)

# pset.addEphemeralConstant(lambda: random.uniform(-1, 1))
# pset.addEphemeralConstant(lambda: random.randint(-10, 10), int)

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin,
               pset=pset)


toolbox = base.Toolbox()
toolbox.register("expr", gp.genFull, pset=pset, min_=1, max_=3)
toolbox.register("individual", tools.initIterate, creator.Individual,
                 toolbox.expr)
# Generate a random tree
tree = gp.genFull(pset=pset, min_=1, max_=3)
tree1=gp.PrimitiveTree(tree)
print(tree1)
# res=gp.compile(tree1, pset)
# print(res)

function = gp.compile(tree1, pset)
print(function(1, 2))
