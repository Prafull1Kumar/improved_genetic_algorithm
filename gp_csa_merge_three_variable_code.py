import operator
import random
import numpy as np
from deap import gp, creator, base, tools,algorithms
import matplotlib.pyplot as plt

# Define the primitive set of functions and terminals
pset_csa = gp.PrimitiveSet("MAIN", arity=3)
pset_csa.addPrimitive(operator.add, arity=2)
pset_csa.addPrimitive(operator.mul, arity=2)
pset_csa.addPrimitive(operator.neg, arity=1)
pset_csa.addTerminal(2.0, name="two")
pset_csa.addTerminal(3.0, name="three")
pset_csa.addTerminal(5.0, name="five")
pset_csa.addTerminal(7.0, name="seven")
pset_csa.addTerminal(11.0, name="eleven")
pset_csa.addTerminal(13.0, name="thirteen")

# add(4,mul(x,8))

# pset.addTerminal(random.randint(0, 9), name="randint")
pset_csa.renameArguments(ARG0="x")
pset_csa.renameArguments(ARG1="y")
pset_csa.renameArguments(ARG2="z")

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

psetz = gp.PrimitiveSet("MAIN", arity=1)
psetz.addPrimitive(operator.add, arity=2)
psetz.addPrimitive(operator.mul, arity=2)
psetz.addPrimitive(operator.neg, arity=1)

psetz.renameArguments(ARG0="z")


pset_st_gp = gp.PrimitiveSet("MAIN", arity=3)
pset_st_gp.addPrimitive(operator.add, arity=2)
pset_st_gp.addPrimitive(operator.mul, arity=2)
pset_st_gp.addPrimitive(operator.neg, arity=1)
pset_st_gp.addTerminal(2.0, name="two")
pset_st_gp.addTerminal(3.0, name="three")
pset_st_gp.addTerminal(5.0, name="five")
pset_st_gp.addTerminal(7.0, name="seven")
pset_st_gp.addTerminal(11.0, name="eleven")
pset_st_gp.addTerminal(13.0, name="thirteen")
pset_st_gp.renameArguments(ARG0="x")
pset_st_gp.renameArguments(ARG1="y")
pset_st_gp.renameArguments(ARG2="z")

cxpb=0.5
mutpb=0.4
awareness_probability=0.2
flight_length=0.8 #(0-1)
randomness=0.8
tournmentsize=8
mintreesize=2
maxtreesize=3
num_runs = 10
solutions_found_cs=0
solutions_found_sta_gp=0
avg_generations_cs=0
avg_deviations_cs=0
avg_generations_sta_gp=0
avg_deviations_sta_gp=0
total_generation=200
population_size=500

# add(one ,y)
def mutUniformx(individual, expr, pset):
    nonxindexes=[]
    j=0
    for node in individual:
        if isinstance(node, gp.Terminal):
            # print(node.value)
            if node.value=='y':
                nonxindexes.append(j)
            if node.value=='z':
                nonxindexes.append(j)    
        j=j+1   
    # print(yindexes)         
    index = random.choice([k for k in range(0, len(individual)-1) if k not in nonxindexes])
    # index = random.randrange(len(individual))
    slice_ = individual.searchSubtree(index)
    type_ = individual[index].ret
    individual[slice_] = expr(pset=pset, type_=type_)
    return individual

def mutUniformy(individual, expr, pset):
    nonyindexes=[]
    j=0
    for node in individual:
        if isinstance(node, gp.Terminal):
            # print(node.value)
            if node.value=='x':
                nonyindexes.append(j)
            if node.value=='z':
                nonyindexes.append(j)    
        j=j+1   
    # print(xindexes)         
    index = random.choice([k for k in range(0, len(individual)-1) if k not in nonyindexes])

    # index = random.randrange(len(individual))
    slice_ = individual.searchSubtree(index)
    type_ = individual[index].ret
    individual[slice_] = expr(pset=pset, type_=type_)
    return individual

def mutUniformz(individual, expr, pset):
    nonzindexes=[]
    j=0
    for node in individual:
        if isinstance(node, gp.Terminal):
            # print(node.value)
            if node.value=='x':
                nonzindexes.append(j)
            if node.value=='y':
                nonzindexes.append(j)    
        j=j+1   
    # print(xindexes)         
    index = random.choice([k for k in range(0, len(individual)-1) if k not in nonzindexes])

    # index = random.randrange(len(individual))
    slice_ = individual.searchSubtree(index)
    type_ = individual[index].ret
    individual[slice_] = expr(pset=pset, type_=type_)
    return individual


# Define the fitness function
def evaluate_csa(individual):
    # print(individual)
    func = gp.compile(individual, pset_csa)

    error = 0
    for x in range(-10, 11):
        for y in range(-10, 11):
            for z in range(-10, 11):
                try:
                    if func(x,y,z) is None:
                        raise TypeError
                    error += abs(func(x,y,z) - (7*x*x*x+5*y*y+11*z)) # Fitness function
                except:
                    error += 100
    return error,

def manualMutUniform_csa(individual, expr, pset):
    # individual :- add(one,two) len=3 fl=0.7 (0.7*3) =2
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

def evaluate_st_gp(individual):
    # print(individual)
    func = gp.compile(individual, pset_st_gp)

    error = 0
    for x in range(-10, 11):
        for y in range(-10, 11): 
            for z in range(-10, 11): 
                try:
                    if func(x,y,z) is None:
                        raise TypeError
                    # y_true.append((7*x*x*x*x*x+45*y*y*x*x*x+56*y*y+107*x*x*x+67))
                    # y_pred.append(func(x, y))
                    # y_diff.append(abs(func(x, y) - (7*x*x*x*x*x+45*y*y*x*x*x+56*y*y+107*x*x*x+67))**2)
                    error += abs(func(x,y,z) - (7*x*x*x+5*y*y+11*z)) # Fitness function
                except:
                    error += 100
    # error = np.sqrt(y_diff)            
    return error,
# Define the genetic programming algorithm
creator.create("FitnessMin_st_gp", base.Fitness, weights=(-1.0,))
creator.create("Individual_st_gp", gp.PrimitiveTree, fitness=creator.FitnessMin_st_gp,
               pset=pset_st_gp)
toolbox_st_gp = base.Toolbox()
toolbox_st_gp.register("expr_st_gp", gp.genFull, pset=pset_st_gp, min_=2, max_=3)
toolbox_st_gp.register("individual", tools.initIterate, creator.Individual_st_gp,
                 toolbox_st_gp.expr_st_gp)
toolbox_st_gp.register("population_st_gp", tools.initRepeat, list, toolbox_st_gp.individual)
toolbox_st_gp.register("evaluate_st_gp", evaluate_st_gp)
toolbox_st_gp.register("select_st_gp", tools.selTournament, tournsize=tournmentsize)
toolbox_st_gp.register("mate_st_gp", gp.cxOnePoint)
toolbox_st_gp.register("mutate_st_gp", gp.mutUniform, expr=toolbox_st_gp.expr_st_gp, pset=pset_st_gp)


creator.create("FitnessMin_csa", base.Fitness, weights=(-1.0,))
creator.create("Individual_csa", gp.PrimitiveTree, fitness=creator.FitnessMin_csa,
               pset=pset_csa)
toolbox = base.Toolbox()
toolbox.register("expr_csa", gp.genFull, pset=pset_csa, min_=mintreesize, max_=maxtreesize)
toolbox.register("exprx", gp.genFull, pset=psetx, min_=mintreesize, max_=maxtreesize)
toolbox.register("expry", gp.genFull, pset=psety, min_=mintreesize, max_=maxtreesize)
toolbox.register("exprz", gp.genFull, pset=psetz, min_=mintreesize, max_=maxtreesize)
toolbox.register("individual_csa", tools.initIterate, creator.Individual_csa,
                 toolbox.expr_csa)
toolbox.register("population_csa", tools.initRepeat, list, toolbox.individual_csa)
toolbox.register("evaluate_csa", evaluate_csa)
toolbox.register("select_csa", tools.selTournament, tournsize=tournmentsize)
toolbox.register("mate_csa", gp.cxOnePoint)
toolbox.register("mutate_csa",manualMutUniform_csa,expr=toolbox.expr_csa, pset=pset_csa)
toolbox.register("mutatex",mutUniformx,expr=toolbox.exprx, pset=psetx)
toolbox.register("mutatey",mutUniformy,expr=toolbox.expry, pset=psety)
toolbox.register("mutatez",mutUniformz,expr=toolbox.exprz, pset=psetz)


# Evolve the random expression

for run in range(num_runs):
    population = toolbox.population_csa(n=population_size)
    population_st_gp = toolbox_st_gp.population_st_gp(n=population_size)
    fitnesses_csa_y=[]
    fitnesses_st_gp=[]
    generation_x=[]
    generation_solutions_found=0
    generation_solutions_found_cs=0
    generation_solutions_found_sta_gp=0
    for generation in range(total_generation):
    # offspring = algorithms.varAnd(population, toolbox, cxpb=0.5, mutpb=0.2)
        if generation_solutions_found_cs==0:
            offspring = [toolbox.clone(ind) for ind in population]

            for i in range(len(offspring)):
                if random.random() > awareness_probability:    # first crow awareness Probability > second crow           
                    k=random.choice([k for k in range(0,population_size-1) if k!=i])
                    if k<i:
                        offspring[k], offspring[i] = toolbox.mate_csa(offspring[k],offspring[i])
                        del offspring[k].fitness.values, offspring[i].fitness.values
                    else:
                        offspring[i], offspring[k] = toolbox.mate_csa(offspring[i],offspring[k])
                        del offspring[k].fitness.values, offspring[i].fitness.values
                else:
                    offspring[i] = toolbox.mutate_csa(offspring[i])
                    del offspring[i].fitness.values

                cnt_x=0
                cnt_y=0
                cnt_z=0
                tree=offspring[i]
                for node in tree:
                    if isinstance(node, gp.Terminal):
                        # print(node.value)
                        if node.value=='x':
                            cnt_x=cnt_x+1
                        if node.value=='y':
                            cnt_y=cnt_y+1   
                        if node.value=='z':
                            cnt_z=cnt_z+1        

                # print(tree,cnt_x,cnt_y)  
                if cnt_x==0:
                    tree=toolbox.mutatex(tree)
                if cnt_y==0:
                    tree=toolbox.mutatey(tree)
                if cnt_z==0:
                    tree=toolbox.mutatez(tree)
                offspring[i]=tree    

            # for i in range(len(offspring)):
                # cnt_x=0
                # cnt_y=0
                # cnt_z=0
                # tree=offspring[i]
                # for node in tree:
                #     if isinstance(node, gp.Terminal):
                #         # print(node.value)
                #         if node.value=='x':
                #             cnt_x=cnt_x+1
                #         if node.value=='y':
                #             cnt_y=cnt_y+1   
                #         if node.value=='z':
                #             cnt_z=cnt_z+1        

                # # print(tree,cnt_x,cnt_y)  
                # if cnt_x==0:
                #     tree=toolbox.mutatex(tree)
                # if cnt_y==0:
                #     tree=toolbox.mutatey(tree)
                # if cnt_z==0:
                #     tree=toolbox.mutatez(tree)
                # offspring[i]=tree

            fits = toolbox.map(toolbox.evaluate_csa, offspring)
            for ind, fit in zip(offspring, fits):
                ind.fitness.values = fit


            # population.extend(offspring); 
            for i in range(len(offspring)):
                if random.random()>randomness: #0.3
                    if offspring[i].fitness.values[0]<toolbox.evaluate_csa(population[i])[0]:
                        population[i]=offspring[i]
                else:
                    population[i]=offspring[i]
                    
           
        population = toolbox.select_csa(population, k=population_size)
        
        #        Genetic Algorithm code
        # ------------------------------------------------------

        if generation_solutions_found_sta_gp==0:
            offspring_st_gp = [toolbox_st_gp.clone(ind) for ind in population_st_gp]

            # Apply crossover and mutation on the offspring
            for i in range(1, len(offspring_st_gp), 2):
                if random.random() < cxpb:
                    offspring_st_gp[i - 1], offspring_st_gp[i] = toolbox_st_gp.mate_st_gp(offspring_st_gp[i - 1],
                                                                offspring_st_gp[i])
                    del offspring_st_gp[i - 1].fitness.values, offspring_st_gp[i].fitness.values

            for i in range(len(offspring_st_gp)):
                if random.random() < mutpb:
                    offspring_st_gp[i], = toolbox_st_gp.mutate_st_gp(offspring_st_gp[i])
                    del offspring_st_gp[i].fitness.values
                    
            fits = toolbox_st_gp.map(toolbox_st_gp.evaluate_st_gp, offspring_st_gp)
            for ind, fit in zip(offspring_st_gp, fits):
                ind.fitness.values = fit
                
        population_st_gp = toolbox_st_gp.select_st_gp(offspring_st_gp, k=len(population_st_gp))

        best_fitness_st_gp=tools.selBest(population_st_gp, k=1)[0].fitness.values[0]
        # print(best_fitness_st_gp)
        # ------------------------------------------------------


        best_fitness=tools.selBest(population, k=1)[0].fitness.values[0]
        print(generation,best_fitness,best_fitness_st_gp)
        generation_x.append(generation)
        fitnesses_csa_y.append(best_fitness)
        fitnesses_st_gp.append(best_fitness_st_gp)
        if best_fitness == 0 and generation_solutions_found_cs==0:
            generation_solutions_found_cs=1
            solutions_found_cs += 1
            avg_generations_cs += generation + 1 # Record the generation in which the solution is found
            # break
        if best_fitness_st_gp == 0 and generation_solutions_found_sta_gp==0:
            generation_solutions_found_sta_gp=1
            solutions_found_sta_gp += 1
            avg_generations_sta_gp += generation + 1 # Record the generation in which the solution is found
            # break    
        if best_fitness_st_gp==0 and best_fitness==0:
            break

    # plt.scatter(generation_x, fitnesses_csa_y)
    # plt.plot(generation_x, fitnesses_csa_y,generation_x, fitnesses_st_gp)
    # plt.plot(generation_x, fitnesses_csa_y, label = "CSABGA")
    # plt.plot(generx2ation_x, fitnesses_st_gp, label = "GA")
    
    # Add a title to a legend
    # plt.legend(title = "Algorithms")
    # plt.title("7*x*x*x+5*y*y+11*z")
    # plt.xlabel("Generation")
    # plt.ylabel("Fitness value")
    # plt.show()        
    # Print the best individual
    best_cs = tools.selBest(population, k=1)[0]
    best_st_gp = tools.selBest(population_st_gp, k=1)[0]
    # print(gp.stringify(best))
    avg_deviations_cs+=best_cs.fitness.values[0]
    avg_deviations_sta_gp+=best_st_gp.fitness.values[0]

    print(run,best_cs)
    function = gp.compile(best_cs, pset_csa)
    print(function(2,2,2))



print("Number of times solution found in CSA after",num_runs,"runs",solutions_found_cs)
print("Number of times solution found in GA after",num_runs,"runs",solutions_found_sta_gp)
if solutions_found_cs>0:
    avg_generations_cs/=solutions_found_cs
if num_runs-solutions_found_cs>0:
    avg_deviations=avg_deviations_cs/(num_runs)

if solutions_found_sta_gp>0:
    avg_generations_sta_gp/=solutions_found_sta_gp
if num_runs-solutions_found_sta_gp>0:
    avg_deviations=avg_deviations_sta_gp/(num_runs)

print("Avg CSA generation:",avg_generations_cs)
print("Avg CSA Deviation:",avg_deviations_cs)

print("Avg GA generation:",avg_generations_sta_gp)
print("Avg GA Deviation:",avg_deviations_sta_gp)