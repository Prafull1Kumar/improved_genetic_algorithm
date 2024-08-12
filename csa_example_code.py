import numpy as np

# Define the fitness function to be optimized
def fitness_function(x):
    return np.sum(np.square(x))

# Define the Crow Search Algorithm function
def crow_search_algorithm(population_size, dim, max_iter, flight_length, memory_size, awareness_prob):
    # Initialize the population
    population = np.random.uniform(low=-10, high=10, size=(population_size, dim))
    # Initialize the memory of each crow with the best solution found so far
    memory = population.copy()
    # Initialize the fitness of the best solution found so far
    best_fitness = float('inf')
    # Start the iterations
    for i in range(max_iter):
        # Evaluate the fitness of the population
        fitness = np.apply_along_axis(fitness_function, 1, population)
        # Check if a new best solution is found
        if np.min(fitness) < best_fitness:
            best_fitness = np.min(fitness)
            best_solution = population[np.argmin(fitness)]
        # Update the memory of each crow with the best solution found so far
        for j in range(population_size):
            if fitness[j] < np.sum(np.square(memory[j])):
                memory[j] = population[j]
        # Update the position of each crow based on its memory and the awareness probability
        for j in range(population_size):
            r = np.random.uniform()
            if r < awareness_prob:
                # Choose a random crow from the population as the neighbor
                k = np.random.randint(population_size)
                # Calculate the distance between the crow and its neighbor
                distance = np.linalg.norm(population[j] - population[k])
                # Update the position of the crow based on its memory and the neighbor's memory
                population[j] += np.random.uniform(-1, 1, dim) * (memory[j] - memory[k]) / distance
            else:
                # Choose a random direction and fly a random distance
                direction = np.random.uniform(low=-1, high=1, size=dim)
                population[j] += direction * flight_length
        # Clip the position of each crow to the solution space
        population = np.clip(population, -10, 10)
    return best_solution, best_fitness

# Example usage
best_solution, best_fitness = crow_search_algorithm(population_size=50, dim=10, max_iter=100, flight_length=0.1, memory_size=5, awareness_prob=0.8)
print("Best solution:", best_solution)
print("Best fitness:", best_fitness)
