import operator
import random
import json
from deap import gp,creator,base,tools

# Define the primitive set of functions and terminals
pset = gp.PrimitiveSet("MAIN", arity=3)

pset.addPrimitive(operator.add, arity=2)
pset.addPrimitive(operator.sub, arity=2)
pset.addPrimitive(operator.mul, arity=2)
pset.addPrimitive(operator.neg, arity=1)
# pset.addTerminal(random.randint(0, 9), name="randint")
# add
pset.addTerminal(1.0, name="one")
pset.addTerminal(2.0, name="two")
pset.renameArguments(ARG0="x")
pset.renameArguments(ARG1="y")
pset.renameArguments(ARG2="z")
# print(pset.arguments[0])
# print(pset.arguments)

# psetx = gp.PrimitiveSet("MAIN", arity=1)
# psetx.addPrimitive(operator.add, arity=2)
# psetx.addPrimitive(operator.mul, arity=2)
# psetx.addPrimitive(operator.neg, arity=1)
# psetx.addTerminal(3.0, name="three")
# psetx.addTerminal(4.0, name="four")
# psetx.renameArguments(ARG0="x")

# psety = gp.PrimitiveSet("MAIN", arity=1)
# psety.addPrimitive(operator.add, arity=2)
# psety.addPrimitive(operator.mul, arity=2)
# psety.addPrimitive(operator.neg, arity=1)
# psety.addTerminal(3.0, name="three")
# psety.addTerminal(4.0, name="four")
# psety.renameArguments(ARG0="y")

def mutUniformx(individual, expr, pset):
    # xindexes=[]
    # j=0
    # for node in individual:
    #     print(node)
    #     if isinstance(node, gp.Terminal):
    #         if node.value=='x':
    #             xindexes.append(j)
    #     j=j+1
    # print(xindexes)            
    # index = random.choice([k for k in range(0, len(individual)-1) if k not in xindexes])

    index = random.randrange(len(individual))
    slice_ = individual.searchSubtree(index)
    type_ = individual[index].ret
    individual[slice_] = expr(pset=pset, type_=type_)
    return individual

def evaluate1(self, arg1, arg2):
    if self == 'add':
        return arg1 + arg2
    elif self == 'sub':
        return arg1 - arg2
    elif self == 'mul':
        return arg1 * arg2
    

def simplify(ind):
    if isinstance(ind, gp.PrimitiveTree):
        # Simplify the subtrees recursively
        for i in range(len(ind)):
            ind[i] = simplify(ind[i])
        # Remove any nodes that do not affect the output
        if len(ind) == 2 and isinstance(ind[0], gp.Terminal) and isinstance(ind[1], gp.Terminal):
            # Two consecutive terminal nodes can be replaced by their product
            return gp.Terminal(ind[0].value * ind[1].value)
        elif len(ind) == 2 and isinstance(ind[1], gp.Terminal) and ind[0].arity == 1:
            # A unary operator with a constant argument can be replaced by its result
            return gp.Terminal(ind[0].evaluate(ind[1].value))
        elif len(ind) == 3 and all(isinstance(ind[i], gp.Terminal) for i in range(1, 3)) and ind[0].arity == 2:
            # A binary operator with two constant arguments can be replaced by its result
            return gp.Terminal(ind[0].evaluate1(ind[1].value, ind[2].value))
        else:
            return ind
    else:
        return ind

def mutUniformy(individual, expr, pset):
    yindexes=[]
    j=0
    for node in individual:
        print(node)
        if isinstance(node, gp.Terminal):
            if node.value=='y':
                yindexes.append(j)
        j=j+1   
    print(yindexes)         
    index = random.choice([k for k in range(0, len(individual)-1) if k not in yindexes])

    # index = random.randrange(len(individual))
    slice_ = individual.searchSubtree(index)
    type_ = individual[index].ret
    individual[slice_] = expr(pset=pset, type_=type_)
    return individual

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin,
               pset=pset)


toolbox = base.Toolbox()
toolbox.register("expr", gp.genFull, pset=pset, min_=1, max_=3)
# toolbox.register("exprx", gp.genFull, pset=psetx, min_=1, max_=3)
# toolbox.register("expry", gp.genFull, pset=psety, min_=1, max_=3)
toolbox.register("individual", tools.initIterate, creator.Individual,
                 toolbox.expr)
# toolbox.register("mutate", gp.mutNodeReplacement, pset=pset)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr, pset=pset)
# toolbox.register("mutatex",mutUniformx,expr=toolbox.exprx, pset=psetx)
# toolbox.register("mutatey",mutUniformy,expr=toolbox.expry, pset=psety)
# Generate a random tree
tree = gp.genFull(pset=pset, min_=1, max_=3)
# print(tree)
printable_tree=gp.PrimitiveTree(tree)
print(printable_tree)
# cnt_x=0
# cnt_y=0
# for node in printable_tree:
#     if isinstance(node, gp.Terminal):
#         print(node.value)
#         if node.value=='x':
#             cnt_x=cnt_x+1
#         if node.value=='y':
#             cnt_y=cnt_y+1    

# print(cnt_x,cnt_y)  
# if cnt_x==0:
#     printable_tree=toolbox.mutatex(printable_tree)
# if cnt_y==0:
#     printable_tree=toolbox.mutatey(printable_tree)

# print(printable_tree)
# modified_printable_tree=gp.PrimitiveTree(printable_tree)
# print(modified_printable_tree)
function = gp.compile(printable_tree, pset)
print(function(1, 2,3))
simplified_exp=simplify(printable_tree)
print(simplified_exp)
# print("Number of arguments:", function.)
# print('Type',printable_tree[0][0].ret)
# mutate_tree= toolbox.mutate(printable_tree)
# for i in range(len(mutate_tree)):
#     printable_mutate_tree=gp.PrimitiveTree(mutate_tree[i])
#     # print(printable_mutate_tree[0])
#     print(printable_mutate_tree,len(printable_mutate_tree))


