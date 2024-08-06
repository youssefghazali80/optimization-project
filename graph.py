import matplotlib.pyplot as plt
import numpy as np
# plot the convergence rate of both the sa and the ga for 100 iteration
Ga_fitness_track = np.loadtxt("Ga_fitness_track")
sa_fitness_track = np.loadtxt("sa_fitness_track")
PSO_fitness_track = np.loadtxt("PSO_fitness_track")
TLBO_fitness_track = np.loadtxt("TLBO_fitness_track")
#plt.plot(range(100), Ga_fitness_track[:100], label="GA")
#plt.plot(range(100), sa_fitness_track[0:100],label='SA')
#plt.plot(range(100),PSO_fitness_track[0:100],label="PSO")
plt.plot(range(100),TLBO_fitness_track[500:600],label="TLBO")
#plt.plot(range(100),PSO_fitness_track[2100:2200],label="w=0.792")
plt.xlabel('iterations')
plt.ylabel('objective_function ')
plt.ylim()
plt.title('convergence of the TLBO algorithm during 100 iterations ')
plt.legend()
plt.grid(True)
plt.show()
# plot_the objective function of 20 runs for for the genetic algorithm and the sa
Ga_fitnesses = np.loadtxt("Ga_best_fitness")
sa_fitnesses = np.loadtxt("Sa_best_fitness")
PSO_fitnesses =np.loadtxt("PSO_best_fitness")
TLBO_fitnesses =np.loadtxt("TLBO_best_fitness")
plt.scatter(range(20), Ga_fitnesses[:20],label="GA")
plt.scatter(range(20), sa_fitnesses[:20] ,label='SA')
plt.scatter(range(20),PSO_fitnesses[:20],label="PSO")
plt.scatter(range(20),TLBO_fitnesses[:20],label="TLBO")

plt.xlabel('Run number')
plt.ylabel('objective_function ')
plt.title('the solution that each algorithm reach for 20 runs')
plt.legend()
plt.grid(True)
plt.show()
print(f"the mean value of the Ga results: {np.mean(Ga_fitnesses)},\n"
      f"the standard deviation for the Ga results: {np.std(Ga_fitnesses)}")
print(f"the mean value of the sa results: {np.mean(sa_fitnesses)},\n"
      f"the standard deviation for the sa results: {np.std(sa_fitnesses)}")
print(f"the mean value of the PSO results: {np.mean(PSO_fitnesses)},\n"
      f"the standard deviation for the PSO results: {np.std(PSO_fitnesses)}")
print(f"the mean value of the TLBO results: {np.mean(TLBO_fitnesses)},\n"
      f"the standard deviation for the TLBO results: {np.std(TLBO_fitnesses)}")
# print the best solution after 20 runs for the ga
best_member_fitness_value = np.argpartition(Ga_fitnesses.flatten(),1 )[:1]

Ga_best_solutions = np.loadtxt("Ga_best_solutions")
#get the  best solution after 20 runs and its objective function
print(f"the best solution for the Ga is : \nnumber of wind turbines :{Ga_best_solutions[3*best_member_fitness_value],} \n"
      f"number of pv panels :{Ga_best_solutions[3*best_member_fitness_value+1]} \n"
      f"the number of batteries :{Ga_best_solutions[3*best_member_fitness_value+2]} \n" 
      f"and the objective function = {Ga_fitnesses[best_member_fitness_value]} ")
#print the best solution after 20 runs for the sa
sa_best_solutions = np.loadtxt("sa_best_solutions")
best_member_fitness_value = np.argpartition(sa_fitnesses.flatten(),1 )[:1]
#get the  best solution after 20 runs and its objective function
print(f"the best solution for the sa is : \nnumber of wind turbines :{sa_best_solutions[3*best_member_fitness_value],} \n"
      f"number of pv panels :{sa_best_solutions[3*best_member_fitness_value+1]} \n"
      f"the number of batteries :{sa_best_solutions[3*best_member_fitness_value+2]} \n"
      f"and the objective function = {sa_fitnesses[best_member_fitness_value]} ")
best_member_fitness_value = np.argpartition(PSO_fitnesses.flatten(),1 )[:1]
PSO_best_solutions = np.loadtxt("PSO_best_solutions")
#get the  best solution after 20 runs and its objective function
print(f"the best solution for the PSO is : \nnumber of wind turbines :{PSO_best_solutions[3*best_member_fitness_value],} \n"
      f"number of pv panels :{PSO_best_solutions[3*best_member_fitness_value+1]} \n"
      f"the number of batteries :{PSO_best_solutions[3*best_member_fitness_value+2]} \n" 
      f"and the objective function = {PSO_fitnesses[best_member_fitness_value]} ")



best_member_fitness_value = np.argpartition(PSO_fitnesses.flatten(),1 )[:1]
TLBO_best_solutions = np.loadtxt("TLBO_best_solutions")
#get the  best solution after 20 runs and its objective function
print(f"the best solution for the TLBO is : \nnumber of wind turbines :{TLBO_best_solutions[3*best_member_fitness_value],} \n"
      f"number of pv panels :{TLBO_best_solutions[3*best_member_fitness_value+1]} \n"
      f"the number of batteries :{TLBO_best_solutions[3*best_member_fitness_value+2]} \n" 
      f"and the objective function = {TLBO_fitnesses[best_member_fitness_value]} ")

plt.show()

