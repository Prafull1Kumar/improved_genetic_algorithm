import operator
import random
import numpy as np
from deap import gp, creator, base, tools,algorithms

# Define the primitive set of functions and terminals
pset = gp.PrimitiveSet("MAIN", arity=2)
pset.addPrimitive(operator.add, arity=2)
pset.addPrimitive(operator.mul, arity=2)
pset.addPrimitive(operator.neg, arity=1)
pset.addTerminal(2.0, name="two")
pset.addTerminal(3.0, name="three")
pset.addTerminal(5.0, name="five")
pset.addTerminal(7.0, name="seven")
pset.addTerminal(11.0, name="eleven")
pset.addTerminal(13.0, name="thirteen")

# add(4,mul(x,8))

# pset.addTerminal(random.randint(0, 9), name="randint")
pset.renameArguments(ARG0="x")
pset.renameArguments(ARG1="y")
# pset.renameArguments(ARG2="z")

psetx = gp.PrimitiveSet("MAIN", arity=1)
psetx.addPrimitive(operator.add, arity=2)
psetx.addPrimitive(operator.mul, arity=2)
psetx.addPrimitive(operator.neg, arity=1)
# psetx.addTerminal(3.0, name="three")
# psetx.addTerminal(4.0, name="four")
psetx.renameArguments(ARG0="x")

psety = gp.PrimitiveSet("MAIN", arity=1)
psety.addPrimitive(operator.add, arity=2)
psety.addPrimitive(operator.mul, arity=2)
psety.addPrimitive(operator.neg, arity=1)
# psety.addTerminal(3.0, name="three")
# psety.addTerminal(4.0, name="four")
psety.renameArguments(ARG0="y")


# 3*x*x*x+4*y*y
# 0.7 0.7 0.3 29/50 #
# 0.7 0.7 0.6 28/50 
# 0.5 0.5 0.6 33/50
# 0.5 0.5 0.7 31/50,(29/50,18479)
# 0.5 0.5 0.5 42/50,38/50,37/50,34/50,30/50,41/50,39/50,(34/50,33120)
# 0.5 0.6 0.5 36/50
# 0.5 0.7 0.5 33/50,34/50,35/50,(35/50,34393)
# 0.3 0.7 0.5 34/50,37/50,(37/50,19028),(43/50,27935),(36/50)
# 0.3 0.8 0.7 (35/50,14578),(36,24730),(40/50,25525)
# 0.2 0.8 0.8 (30/50,10007),(28/50,15900),(29/50,10758)

# 3*x*x*x*4*y*y
# 0.3 0.7 0.5 32/50,33/50
# 0.5 0.5 0.5 32/50
# 0.3 0.8 0.7 (32/50,5231098)

# x
# 0.5 0.5 0.5 50/50

# 4*x*x+7*x
# 0.5 0.5 0.5 36/50
# 0.3 0.7 0.5 43/50,41/50

# 4*x*x+7*x+13
# 0.3 0.7 0.5 34/50,31/50,35/50

# 6*x*x*x+4*x*x+7*x+13 bahot kharab 
awareness_probability=0.3
flight_length=0.8 #(0-1)
randomness=0.7
tournmentsize=4
mintreesize=2
maxtreesize=3
num_runs = 50
solutions_found=0
avg_generations=0
avg_deviations=0

# add(one ,y)
def mutUniformx(individual, expr, pset):
    yindexes=[]
    j=0
    for node in individual:
        if isinstance(node, gp.Terminal):
            # print(node.value)
            if node.value=='y':
                yindexes.append(j)
        j=j+1   
    # print(yindexes)         
    index = random.choice([k for k in range(0, len(individual)-1) if k not in yindexes])
    # index = random.randrange(len(individual))
    slice_ = individual.searchSubtree(index)
    type_ = individual[index].ret
    individual[slice_] = expr(pset=pset, type_=type_)
    return individual

def mutUniformy(individual, expr, pset):
    xindexes=[]
    j=0
    for node in individual:
        if isinstance(node, gp.Terminal):
            # print(node.value)
            if node.value=='x':
                xindexes.append(j)
        j=j+1   
    # print(xindexes)         
    index = random.choice([k for k in range(0, len(individual)-1) if k not in xindexes])

    # index = random.randrange(len(individual))
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
                error += abs(func(x, y) - (3*x*x*x*4*y*y)) # Fitness function
            except:
                error += 100
    return error,

def manualMutUniform(individual, expr, pset):

    n=int(flight_length*len(individual)) #(0-m) m be the length of individual 10*0.5 =5
    indexs=random.sample(range(0, len(individual)), n)
    # index = random.randrange(len(individual))
    # print(flight_length,len(individual),n)
    # print(indexs)
    for index in indexs:
        # print(index)
        if index<len(individual):
            slice_ = individual.searchSubtree(index)
            type_ = individual[index].ret
            # print(individual,index,slice_,type_)
            individual[slice_] = expr(pset=pset, type_=type_)
    return individual
# Define the genetic programming algorithm
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin,
               pset=pset)
toolbox = base.Toolbox()
toolbox.register("expr", gp.genFull, pset=pset, min_=mintreesize, max_=maxtreesize)
toolbox.register("exprx", gp.genFull, pset=psetx, min_=2, max_=3)
toolbox.register("expry", gp.genFull, pset=psety, min_=2, max_=3)
toolbox.register("individual", tools.initIterate, creator.Individual,
                 toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("evaluate", evaluate)
toolbox.register("select", tools.selTournament, tournsize=tournmentsize)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("mutate",manualMutUniform,expr=toolbox.expr, pset=pset)
toolbox.register("mutatex",mutUniformx,expr=toolbox.exprx, pset=psetx)
toolbox.register("mutatey",mutUniformy,expr=toolbox.expry, pset=psety)
# toolbox.register("mutate", gp.mutNodeReplacement, pset=pset)
# toolbox.register("mutate", tools.mutPolynomialBounded, eta=1.0, low=-1.0, up=1.0, indpb=0.1)
# toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr, pset=pset)

# Evolve the random expression

for run in range(num_runs):
    population = toolbox.population(n=100)
    for generation in range(100):
    # offspring = algorithms.varAnd(population, toolbox, cxpb=0.5, mutpb=0.2)
        offspring = [toolbox.clone(ind) for ind in population]
        
        for i in range(len(offspring)):
            if random.random() > awareness_probability:    # first crow awareness Probability > second crow           
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
                    # print(node.value)
                    if node.value=='x':
                        cnt_x=cnt_x+1
                    if node.value=='y':
                        cnt_y=cnt_y+1    

            # print(tree,cnt_x,cnt_y)  
            if cnt_x==0:
                tree=toolbox.mutatex(tree)
            if cnt_y==0:
                tree=toolbox.mutatey(tree)
            offspring[i]=tree

        fits = toolbox.map(toolbox.evaluate, offspring)
        for ind, fit in zip(offspring, fits):
            ind.fitness.values = fit


        # population.extend(offspring); 
        for i in range(len(offspring)):
            if random.random()>randomness:
                if offspring[i].fitness.values[0]<toolbox.evaluate(population[i])[0]:
                    population[i]=offspring[i]
            else:
                population[i]=offspring[i]
                
           
        population = toolbox.select(population, k=100)
        
        best_fitness=tools.selBest(population, k=1)[0].fitness.values[0]
        # print(best_fitness)
        if best_fitness == 0:
            solutions_found += 1
            avg_generations += generation + 1 # Record the generation in which the solution is found
            break

    # Print the best individual
    best = tools.selBest(population, k=1)[0]
    # print(gp.stringify(best))
    avg_deviations+=best.fitness.values[0]
    print(run,best)
    function = gp.compile(best, pset)
    print(function(2, 2))



print("Number of times solution found after",num_runs,"runs",solutions_found)
if solutions_found>0:
    avg_generations/=solutions_found
if num_runs-solutions_found>0:
    avg_deviations=avg_deviations/(num_runs-solutions_found)

print("Avg generation:",avg_generations)
print("Avg Deviation:",avg_deviations)