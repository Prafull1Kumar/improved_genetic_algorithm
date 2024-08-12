import operator
import random
import json
from deap import gp,creator,base,tools
# Define primitive set and add function
pset = gp.PrimitiveSet("MAIN", arity=2)
pset.addPrimitive(operator.add, arity=2)
pset.addTerminal(1.0, name="one")
pset.addTerminal(2.0, name="two")
pset.renameArguments(ARG0="x")
pset.renameArguments(ARG1="y")

# Define tree structure for add function
tree = gp.PrimitiveTree.from_string("(add x y)", pset)
# print(tree.fitness)
# Evaluate tree with values for x and y
x = 3
y = 4
# add_result = tree.compile(x, y)
# modified_printable_tree=gp.PrimitiveTree(printable_tree)
print(tree)
function = gp.compile(tree, pset)

print(function(x,y))  # Output: 7
tree2=gp.PrimitiveTree.from_string("(add "+str(tree)+" y)",pset)
print(tree2)
function1=gp.compile(tree2,pset)
print(function1(x,y))