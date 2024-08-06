
import numpy as np
# generate array that stores best solution  for PSO algortihm
PSO_best_solution =[]
filename = "PSO_best_solutions"
np.savetxt(filename, PSO_best_solution, delimiter='\n')


#generate an array that stores the fitness_values of the best solutions for PSO algorithm
PSO_best_fitness =[]
filename = "PSO_best_fitness"
np.savetxt(filename, PSO_best_fitness, delimiter='\n')


# generate array that tracks the objective function for the PSO algorithm to see the convergence rate
PSO_fitness_track =[]
filename = "PSO_fitness_track"
np.savetxt(filename, PSO_fitness_track, delimiter='\n')