import operator
import random
import numpy as np
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

psetx = gp.PrimitiveSet("MAIN", arity=1)
psetx.addPrimitive(operator.add, arity=2)
psetx.addPrimitive(operator.mul, arity=2)
psetx.addPrimitive(operator.neg, arity=1)
psetx.renameArguments(ARG0="x")

psety = gp.PrimitiveSet("MAIN", arity=1)
psety.addPrimitive(operator.add, arity=2)
psety.addPrimitive(operator.mul, arity=2)
psety.addPrimitive(operator.neg, arity=1)
psety.renameArguments(ARG0="y")

awareness_probability=0.3
flight_length=0.5 #(0-1)
randomness=0.5
tournmentsize=4
mintreesize=2
maxtreesize=3


def mutUniformx(individual, expr, pset):
    # yindexes=[]
    # j=0
    # for node in individual:
    #     if isinstance(node, gp.Terminal):
    #         # print(node.value)
    #         if node.value=='y':
    #             yindexes.append(j)
    #     j=j+1   
    # print(yindexes)         
    # index = random.choice([k for k in range(0, len(individual)-1) if k not in yindexes])
    index = random.randrange(len(individual))
    slice_ = individual.searchSubtree(index)
    type_ = individual[index].ret
    individual[slice_] = expr(pset=pset, type_=type_)
    return individual

def mutUniformy(individual, expr, pset):
    # xindexes=[]
    # j=0
    # for node in individual:
    #     if isinstance(node, gp.Terminal):
    #         # print(node.value)
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
                error += abs(func(x, y) - (3*x*x*x + 4*y*y)) # Fitness function
            except:
                error += 100
    return error,

def manualMutUniform(individual, expr, pset):

    n=int(flight_length*len(individual)) #(0-m) m be the length of individual
    indexs=random.sample(range(0, len(individual)), n)
    
    for index in indexs:
        
        if index<len(individual):
            slice_ = individual.searchSubtree(index)
            type_ = individual[index].ret
            
            individual[slice_] = expr(pset=pset, type_=type_)
    return individual
# Define the genetic programming algorithm
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin,
               pset=pset)
toolbox = base.Toolbox()
toolbox.register("expr", gp.genFull, pset=pset, min_=mintreesize, max_=maxtreesize)
toolbox.register("exprx", gp.genFull, pset=psetx, min_=mintreesize, max_=maxtreesize)
toolbox.register("expry", gp.genFull, pset=psety, min_=mintreesize, max_=maxtreesize)
toolbox.register("individual", tools.initIterate, creator.Individual,
                 toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("evaluate", evaluate)
toolbox.register("select", tools.selTournament, tournsize=tournmentsize)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("mutate",manualMutUniform,expr=toolbox.expr, pset=pset)
toolbox.register("mutatex",mutUniformx,expr=toolbox.exprx, pset=psetx)
toolbox.register("mutatey",mutUniformy,expr=toolbox.expry, pset=psety)

population = toolbox.population(n=100)
for generation in range(100):

    offspring = [toolbox.clone(ind) for ind in population]

    for i in range(len(offspring)):
        if random.random() < awareness_probability:
            
            k=random.choice([k for k in range(0,99) if k!=i])
            
            if k<i:
                offspring[k], offspring[i] = toolbox.mate(offspring[k],offspring[i])
                del offspring[k].fitness.values, offspring[i].fitness.values
            else:
                offspring[i], offspring[k] = toolbox.mate(offspring[i],offspring[k])
                del offspring[k].fitness.values, offspring[i].fitness.values
            
        else:
            
            offspring[i] = toolbox.mutate(offspring[i])
            del offspring[i].fitness.values

    for i in range(len(offspring)):
        cnt_x=0
        cnt_y=0
        tree=offspring[i]
        for node in tree:
            if isinstance(node, gp.Terminal):
                if node.value=='x':
                    cnt_x=cnt_x+1
                if node.value=='y':
                    cnt_y=cnt_y+1    
        if cnt_x==0:
            tree=toolbox.mutatex(tree)
        if cnt_y==0:
            tree=toolbox.mutatey(tree)
        offspring[i]=tree

    fits = toolbox.map(toolbox.evaluate, offspring)
    for ind, fit in zip(offspring, fits):
        ind.fitness.values = fit
        

    for i in range(len(offspring)):
        if random.random()>randomness:
            if offspring[i].fitness.values[0]<toolbox.evaluate(population[i])[0]:
                population[i]=offspring[i]
        else:
            population[i]=offspring[i]
            
        
    population = toolbox.select(population, k=100)
 
   
    best_fitness=tools.selBest(population, k=1)[0].fitness.values[0]
    print(best_fitness)
    

# Print the best individual
best = tools.selBest(population, k=1)[0]
# print(gp.stringify(best))
print(best)
function = gp.compile(best, pset)
print(function(1, 2))
