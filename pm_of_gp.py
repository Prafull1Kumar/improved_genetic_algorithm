import operator
import random
from deap import gp, creator, base, tools,algorithms

# Define the primitive set of functions and terminals
pset_st_gp = gp.PrimitiveSet("MAIN", arity=1)
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
# pset_st_gp.renameArguments(ARG1="y")
# pset_st_gp.renameArguments(ARG2="z")
# pset.renameArguments(ARG1="y")
cxpb=0.5
mutpb=0.4
num_runs = 1
solutions_found=0
avg_generations=0
avg_deviations=0
total_generation=200
population_size=500
tournmentsize=8
# Define the fitness function
def evaluate_st_gp(individual):
    # print(individual)
    func = gp.compile(individual, pset_st_gp)

    error = 0
    for x in range(-10, 11):
        # for y in range(-10, 11): 
        #     for z in range(-10, 11): 
        try:
            if func(x) is None:
                raise TypeError
            # y_true.append((7*x*x*x*x*x+45*y*y*x*x*x+56*y*y+107*x*x*x+67))
            # y_pred.append(func(x, y))
            # y_diff.append(abs(func(x, y) - (7*x*x*x*x*x+45*y*y*x*x*x+56*y*y+107*x*x*x+67))**2)
            error += abs(func(x) - (x*x*x*x+4*x*x+7*x+11)) # Fitness function
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

# Evolve the random expression
for run in range(num_runs):
    population_st_gp = toolbox_st_gp.population_st_gp(n=population_size)
    generation_solutions_found=0
    for generation in range(total_generation):
        # offspring = algorithms.varAnd(population, toolbox, cxpb=0.5, mutpb=0.2)
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
        # population=offspring
        # print(tools.selBest(population, k=1)[0].fitness.values[0])
        best_fitness_st_gp=tools.selBest(population_st_gp, k=1)[0].fitness.values[0]
        print(best_fitness_st_gp)
        if best_fitness_st_gp == 0 and generation_solutions_found==0:
            generation_solutions_found=1
            solutions_found += 1
            avg_generations += generation + 1 # Record the generation in which the solution is found
            # break

    # Print the best individual
    best = tools.selBest(population_st_gp, k=1)[0]
    # print(gp.stringify(best))
    avg_deviations+=best.fitness.values[0]
    print(best)
    function = gp.compile(best, pset_st_gp)
    print(function(2))

print("Number of times solution found after",num_runs,"runs",solutions_found)
if solutions_found>0:
    avg_generations/=solutions_found
if num_runs-solutions_found>0:
    avg_deviations=avg_deviations/(num_runs-solutions_found)

print("Avg generation:",avg_generations)
print("Avg Deviation:",avg_deviations)
