import numpy as np
# generate array that stores best solution  for PSO algortihm
TLBO_best_solution =[]
filename = "TLBO_best_solutions"
np.savetxt(filename, TLBO_best_solution, delimiter='\n')


#generate an array that stores the fitness_values of the best solutions for PSO algorithm
TLBO_best_fitness =[]
filename = "TLBO_best_fitness"
np.savetxt(filename, TLBO_best_fitness, delimiter='\n')


# generate array that tracks the objective function for the PSO algorithm to see the convergence rate
TLBO_fitness_track =[]
filename = "TLBO_fitness_track"
np.savetxt(filename, TLBO_fitness_track, delimiter='\n')