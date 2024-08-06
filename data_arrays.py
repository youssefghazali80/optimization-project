#define Ga best solution for each iteration    store genetic algorithm best solutions
import numpy as np
# generate array that stores best solution  for genetic algortihm for gentic algorithm
'''
Ga_best_solution =[]
filename = "Ga_best_solutions"
np.savetxt(filename, Ga_best_solution, delimiter='\n')


#generate an array that stores the fitness_values of the best solutions for genetic algorithm
Ga_best_fitness =[]
filename = "Ga_best_fitness"
np.savetxt(filename, Ga_best_fitness, delimiter='\n')


# generate array that tracks the objective function for the Ga_algorithm to see the convergence rate
Ga_fitness_track =[]
filename = "Ga_fitness_track"
np.savetxt(filename, Ga_fitness_track, delimiter='\n')
'''
#generate an array that stores the fitness_values of the best solutions for simulated annealing
sa_best_fitness =[]
filename = "sa_best_fitness"
np.savetxt(filename, sa_best_fitness, delimiter='\n')

#generate array that stores best solution for simulated annealing algortihm for gentic algorithm
sa_best_solution =[]
filename = "sa_best_solutions"
np.savetxt(filename, sa_best_solution, delimiter='\n')
# generate array that tracks the objective function for the simulated annealing to see the convergence rate
sa_fitness_track =[]
filename = "sa_fitness_track"
np.savetxt(filename, sa_fitness_track, delimiter='\n')