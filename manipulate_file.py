import numpy as np
TLBO_best_fitness = np.loadtxt("TLBO_best_fitness")



TLBO_best_fitness = np.append(TLBO_best_fitness,TLBO_best_fitness);
TLBO_best_fitness = np.append(TLBO_best_fitness,TLBO_best_fitness);
np.savetxt("TLBO_best_fitness", TLBO_best_fitness, delimiter='\n')