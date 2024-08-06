import random
import numpy as np
import time
import math
import matplotlib.pyplot as plt
# initializing the initial values of the specs of the wind turbine
Pr= 10 *(10**3)   # rated power of each turbine KW
Zhub =15    #   height of the hub of the wind turbine
filename = "wind_speed_data.txt"
Vanem= np.loadtxt(filename)  # wind velocity at the height of the anemometer
Zanem = 10  # height of the anemometer
y = 0.14   # the hellman  exponent
Vhub = (Vanem) * ((Zhub / Zanem)**y)
Vr = 12     # rated speed of the wind
Vi = 3     # cut in speed for the wind
Vo = 25    # cut off speed
def calculate_power_single_turbine(Vhub,Pr,Vo) :
    Pt= np.empty(8760)
    for i in range(len(Vhub)):

        if (Vhub[i]>=Vo or Vhub[i]<=Vi):
            Pt[i] =0
        elif(Vhub[i] >Vi and Vhub[i]<Vr):
            Pt[i] = Pr*(((Vhub[i]**2)-(Vi**2))/((Vr**2)-(Vi**2)))
        else:
            Pt[i] =Pr
    return Pt
Pt = calculate_power_single_turbine(Vhub,Pr,Vo)
icw = 5050   # initial cost of the windturbine
rcw = 5050   # replacement cost of the wind turbine
# initializing the initial value of the pv panels
PPV_STC = 265  # Rated power of PV module (W)
Fpv = 0.95  # Derating factor (e.g., 95%)
G_t = np.loadtxt("solar_radiation_data.txt")  # Instantaneous global solar radiation incident (W/m^2)
GT_STC = 1000  # Standard radiation (W/m^2)
alpha_p = -0.004  # Temperature coefficient of maximum power
TSTC = 25  # Cell temperature at standard test conditions (°C)
filename = "temperature_data.txt"
Ta_t = np.loadtxt(filename)  # Ambient temperature at time t (°C)
NOCT = 45  # Normal operating cell temperature (°C)
def pv_panel_power(PPV_STC, Fpv, G_t, GT_STC, alpha_p, TSTC, Ta_t, NOCT):
    PPV_t=np.empty(8760)
    Tc_t = np.empty(8760)
    for i in range(len(Ta_t)) :
        Tc_t[i] = Ta_t[i] + ((NOCT - 20) / 800) * G_t[i]
        PPV_t[i] = PPV_STC * Fpv * (G_t[i] / GT_STC) * (1 + alpha_p * (Tc_t[i] - TSTC))
    PPV_t =np.clip(PPV_t,0,265)
    return PPV_t
PPV_t=pv_panel_power(PPV_STC, Fpv, G_t, GT_STC, alpha_p, TSTC, Ta_t, NOCT)
# initializing the initial values and the specs of the battery
Eload = 357*(10**3)    # this is average energy needed from the system KWh/ day
Ndis = 0.86      #discharging efficiency
Nch =  0.86      # charging efficiency
nd = 1      # autounumy days {number of days thet the system can fully depend on charged batterires}
SDR =0.02    # self discharge rate
Dod =  0.8  # maximum depth of discharge
nb = 0.86  # battery overall efficincy
vb = 2     # rated voltage of the battery
Cb = 1000 # rated ampere hour of the battery
SE_batt = Cb*vb
def calculate_maximum_storage_capacity(n_batt , SE_batt):
    return (n_batt*SE_batt)
def calculate_the_minimun_number_batteries(Eload, nd , nb , Dod ,vb ,Cb):
    Cah = (Eload*nd)/ (nb*Dod*vb)
    return (Cah/Cb)



# define the constraints variable
Nwt_min = 1      # minimum number of wind turbines
Nwt_max = 40    #maximum number of wind turbines
Npv_min = 100    # minimum number rof pv panels
Npv_max = 1000   #maximum number of pv panels
Nbatt_min = int(calculate_the_minimun_number_batteries(Eload,nd,nb,Dod,vb,Cb))  #minimum number of batteries
Nbatt_max = 500      #maximum number of batteries
#initialize
'''
Nwt = random.randint(1,20) 
Npv = random.randint(100,1000)
N_batt  = random.randint(Nbatt_min,500)
'''
lpsplim = 0 # loss of power supply index
n_y=20    # 20 years (the lifetime of the project)
d_r=0.06  # discount rate

def update_lpsp(lpsp , Pload , Pw ,Pv ,Pb,power_d):
     lpsp = lpsp +((Pload-(Pw+Pv+Pb))/np.sum(power_d))
     return lpsp
def calclate_surplus_power  (Pload , Pw ,Pv ,Nrec ):
    Pbc = Pv +(Pw-Pload)*Nrec
    return Pbc
def calcualte_power_difference(Pload , Pw ,Pv ,Nrec):
    Pbd= (Pw - Pload) * Nrec -Pv
    return Pbd
def charge_battery (SOC ,Psurp ,Nch ,SDR):
    SOC = SOC*(1-SDR)+Psurp*Nch
    return SOC
def discharge_battery(SOC ,PD ,Ndis ,SDR):
    SOC = SOC * (1 - SDR) -  (PD / Ndis)
    return SOC
def calculate_initial_cost (Nwt,Npv,N_batt):
    IC= Nwt * 5050 + Npv *119 + N_batt *50
    return IC
def calculate_replacement_cost(Nwt,Npv,N_batt,CRF,i):
    RC= CRF  *(((Nwt*5050)/(1+i)**20)+(4* (N_batt*50) /(i+1)**20))
    return RC
def calculate_maintanenece_cost(Nwt,Npv,N_batt):
    ACM = Nwt*10 + N_batt*5
    return ACM
def calculate_annual_cost (Nwt,Npv,N_batt,d_r,n_y):
    CRF = (0.06 * ((0.06 + 1) ** 20)) / (((1 + 0.06) ** 20) - 1)
    IC = calculate_initial_cost(Nwt,Npv,N_batt)
    ARC = calculate_replacement_cost(Nwt,Npv,N_batt,CRF,d_r)
    AMC = calculate_maintanenece_cost(Nwt,Npv ,N_batt)
    return (CRF*IC+ARC+AMC)
def calculate_power_to_the_battery_to_be_full (SOC,SOC_Max):
   return ((SOC_Max-(SOC*(1-0.02)))/0.81)
def calculate_power_from_the_battery_to_be_empty (SOC,SOC_Min):
   return (((SOC*(1-0.02))-SOC_Min)*0.81)
def calculate_dumped_power(Pwt ,Ppv,Pload ,Nrec):
    return(Ppv+(Pwt-Pload)*Nrec)
def check_feasibility(current_solution ,Nwt_min ,Nwt_max,Npv_min,Npv_Max ,Nbatt_Min,Nbatt_Max):
        # Generate a neighboring solution by perturbing the current solution
        wind_turbines, pv_panels, batteries = current_solution
        # Randomly increment or decrement the number of wind turbines, PV panels, and batteries
        new_wind_turbines = wind_turbines  # Random integer between -2 and 2
        new_pv_panels = pv_panels   # Random integer between -10 and 10
        new_batteries = batteries  # Random integer between -10 and 10
        # Ensure that the number of turbines, panels, and batteries doesn't go below zero
        new_wind_turbines = max(new_wind_turbines, Nwt_min)
        new_wind_turbines = min (new_wind_turbines,Nwt_max)
        new_pv_panels = min(new_pv_panels,Npv_Max)
        new_pv_panels = max(new_pv_panels, Npv_min)
        new_batteries = max(new_batteries, Nbatt_Min)
        new_batteries = min(new_batteries,Nbatt_Max)
        return [new_wind_turbines, new_pv_panels, new_batteries]

Power_demand = np.loadtxt("power_demand_data.txt")
def calculate_annual_supplied_load_and_penaltizing_objective_function (Nwt,Npv,N_batt,Pwt,Ppv,SE_batt,lpsp_lim,Pd):
    lpsp=0
    dumped_power= np.zeros(8760)
    asl = []
    total_pwt = Nwt * Pwt
    total_Ppv = Npv * Ppv
    SOC_max = calculate_maximum_storage_capacity(N_batt,SE_batt)
    SOC_min = 0.01 * SOC_max
    SOC = SOC_min
    for i in range(8760):
        #print(f"the power of turbines is {total_Ppv[i]}  and the power of panels is :{total_Ppv[i]} and the power required is {Pd[i]} ")
        #print(total_pwt[i]+total_Ppv[i]-Pd[i])
        if (total_pwt[i]+total_Ppv[i]>Pd[i]):
            Pbc = calclate_surplus_power(Pd[i],total_pwt[i],total_Ppv[i],0.81)
            if (charge_battery(SOC,Pbc,Nch,SDR)>SOC_max):
                SOC=SOC_max
                dumped_power[i]=calculate_dumped_power(total_pwt[i],total_Ppv[i],Pd[i],0.81)

            else :
                SOC =charge_battery(SOC,Pbc,Nch,SDR)
                asl = np.append(asl,Pd[i])


        elif (total_pwt[i]+total_Ppv[i] < Pd[i]):
            Pbd = calcualte_power_difference(Pd[i],total_pwt[i],total_Ppv[i],0.81)
            if (discharge_battery(SOC,-Pbd,Ndis,SDR)<SOC_min):
                if (SOC==SOC_min):
                    pb=0
                else:
                   pb = calculate_power_from_the_battery_to_be_empty(SOC,SOC_min)
                   SOC=SOC_min
                lpsp =  update_lpsp(lpsp,Pd[i],total_pwt[i],total_Ppv[i],pb,Pd)

                asl= np.append(asl,[total_pwt[i]+total_Ppv[i]+pb])
            elif (discharge_battery(SOC,Pbd ,Ndis,SDR)>SOC_min):
                SOC=discharge_battery(SOC, Pbd ,Ndis,SDR)
                asl = np.append(asl, Pd[i])
        else :
            asl = np.append(asl, Pd[i])


    return([np.sum(asl),lpsp,np.sum(dumped_power)])
def objective_function (Nwt,Npv,N_batt):
    Ca= calculate_annual_cost(Nwt,Npv,N_batt,0.05,20)
    [asl,lpsp,power_dumped]=calculate_annual_supplied_load_and_penaltizing_objective_function(Nwt,Npv,N_batt,Pt,PPV_t,SE_batt,lpsplim,Power_demand)
    if (lpsp <= 0):

        return(0.96*(Ca/(asl/1000))+ 10**-11 * power_dumped)

    else :
        return((0.96*(Ca/(asl/1000)) +10**9* lpsp)+  10**-11 * power_dumped)
def generate_initial_solutions(Nwt_min ,Nwt_max,Npv_min,Npv_Max ,Nbatt_Min,Nbatt_Max,pop_size):

    array_wt = np.random.randint(25, Nwt_max, size=(pop_size, 1))
    array_pv = np.random.randint(500, Npv_Max, size=(pop_size, 1))
    array_Nbatt = np.random.randint(Nbatt_Min, Nbatt_Max, size=(pop_size, 1))
    initial_generation = np.concatenate((array_wt,array_pv,array_Nbatt),axis=1)

    return initial_generation
def update_global_best(solution,fitness_values):                               #update global best according to star topology
    (a,b)=solution.shape
    best =fitness_values[0]
    global_best = solution[0,0],solution[0,1],solution[0,2]

    for i in range(a-1):
        new_fitness= fitness_values[i+1]
        if (new_fitness<best):
            global_best = solution[i+1]
            best=new_fitness


    return global_best,best
def update_personal_best(solution,personal_best,fitness_values):
    (a,b)=solution.shape
    for i in range (a):
        if (fitness_values[i]<objective_function(personal_best[i,0],personal_best[i,1],personal_best[i,2])):
            personal_best[i]=solution[i]

    return personal_best


# load the arrays to store the data in it
PSO_best_solutions = np.loadtxt("PSO_best_solutions")
PSO_best_fitness = np.loadtxt("PSO_best_fitness")
PSO_fitness_track = np.loadtxt("PSO_fitness_track")






PSO_best_solutions = np.loadtxt("PSO_best_solutions")
PSO_best_fitness = np.loadtxt("PSO_best_fitness")
PSO_fitness_track = np.loadtxt("PSO_fitness_track")

#defining the pso algorithm
def PSO_alogrithm(Nwt_min ,Nwt_max,Npv_min,Npv_Max ,Nbatt_Min,Nbatt_Max,pop_size,iteration_num,PSO_fitness_track):
    start_time =time.time()
    current_position = generate_initial_solutions(Nwt_min, Nwt_max, Npv_min, Npv_Max, Nbatt_Min, Nbatt_Max, pop_size)   # initialize the first psotion of the particles
    new_velocity = np.random.rand(pop_size, 3) * 0.2    # initialize the initial velocity of the particles
    personal_best = current_position
    w = 0.792        # widely used constant inertia value
    c1 = 1.5  # widely used cognitive factor
    c2 = 1.5    # widely used social factor
    fitness_values = np.zeros(shape=(pop_size))
    for r in range(pop_size):  # calculate initail fitness values of the particles
        fitness_values[r]=objective_function(current_position[r,0],current_position[r,1],current_position[r,2])
    best_solution,best_fitness=update_global_best(current_position,fitness_values) #  initialize the best solutions and the best fitness_values
    for i in range(iteration_num):
        current_velocity = new_velocity
        current_position = current_position+current_velocity    #update the position of the particles
        current_position = np.round(current_position)
        for j in range (pop_size):                            #check the feasibility and update the solution if necessary
            current_position[j] =check_feasibility(current_position[j],Nwt_min,Nwt_max,Npv_min,Npv_Max,Nbatt_Min,Nbatt_Max)

        for q in range (pop_size):   #calculate the fitness_value of the new posistion of the particles
            fitness_values[q] = objective_function(current_position[q, 0], current_position[q, 1], current_position[q, 2])

        global_best,current_best_fitness = update_global_best(current_position,fitness_values)  #update the global best

             #record the convergence of the best  fitness_value
        if(current_best_fitness<best_fitness):
            best_fitness=current_best_fitness
            best_solution=global_best
        PSO_fitness_track = np.append(PSO_fitness_track, [current_best_fitness])

        personal_best = update_personal_best(current_position,personal_best,fitness_values)
        new_velocity = w*current_velocity + c1*(personal_best-current_position)+ c2*(global_best-current_position)  #update velocity of the particles
        print(i)
    end_time =time.time()
    #print(end_time-start_time)



    return best_solution ,best_fitness,PSO_fitness_track








for k in range(20):

    best_solution , best_fitness,PSO_fitness_track = PSO_alogrithm(Nwt_min,Nwt_max,Npv_min,Npv_max,Nbatt_min,Nbatt_max,30,100,PSO_fitness_track)
    print(f"the best solution is :{best_solution}, /n the objective function is {best_fitness}")
    PSO_best_fitness =np.append(PSO_best_fitness,[best_fitness])
    PSO_best_solutions = np.concatenate((PSO_best_solutions,best_solution),axis=0)
    print(k)



#np.savetxt("PSO_best_fitness", PSO_best_fitness, delimiter='\n')
#np.savetxt("PSO_best_solutions", PSO_best_solutions ,delimiter='\n')
np.savetxt("PSO_fitness_track",PSO_fitness_track,delimiter='/n')





