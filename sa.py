import random
import numpy as np
import math
import time
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
print(Pt)
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
print(PPV_t)
# initializing the initial values and the specs of the battery
Eload = 357*(10**3)    # this is average energy needed from the system KWh/ day
Ndis = 0.86      #discharging efficiency
Nch =  0.86      # charging efficiency
nd = 1       # autounumy days {number of days thet the system can fully depend on charged batterires}
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
Nwt_min = 1
Nwt_max = 40
Npv_min = 100
Npv_max = 1000
Nbatt_min = int(calculate_the_minimun_number_batteries(Eload,nd,nb,Dod,vb,Cb))
Nbatt_max = 400
Nwt = random.randint(10,Nwt_max)
Npv = random.randint(400,Npv_max)
N_batt  = random.randint(Nbatt_min,500)
lpsplim = 0.2 # loss of power supply index
n_y=20    # 20 years (the lifetime of the project)
i=0.06  # discount rate
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
def calculate_annual_cost (Nwt,Npv,N_batt,i,n):
    CRF = (i*((i+1)**n))/(((1+i)**n)-1)
    IC = calculate_initial_cost(Nwt,Npv,N_batt)
    ARC = calculate_replacement_cost(Nwt,Npv,N_batt,CRF,i)
    AMC = calculate_maintanenece_cost(Nwt,Npv ,N_batt)
    return (CRF*IC+ARC+AMC)
def calculate_power_to_the_battery_to_be_full (SOC,SOC_Max):
   return ((SOC_Max-(SOC*(1-0.02)))/0.81)
def calculate_power_from_the_battery_to_be_empty (SOC,SOC_Min):
   return (((SOC*(1-0.02))-SOC_Min)*0.81)
def calculate_dumped_power(Pwt ,Ppv,Pload ,Nrec):
    return(Ppv+(Pwt-Pload)*Nrec)
def random_neighbor(current_solution ,Nwt_min ,Nwt_max,Npv_min,Npv_Max ,Nbatt_Min,Nbatt_Max):
        # Generate a neighboring solution by perturbing the current solution
        wind_turbines, pv_panels, batteries = current_solution
        # Randomly increment or decrement the number of wind turbines, PV panels, and batteries
        new_wind_turbines = wind_turbines + random.randint(-5, 5)  # Random integer between -2 and 2
        new_pv_panels = pv_panels + random.randint(-60, 60)  # Random integer between -10 and 10
        new_batteries = batteries + random.randint(-50, 50)  # Random integer between -10 and 10
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
    Ca= calculate_annual_cost(Nwt,Npv,N_batt,i,n_y)
    [asl,lpsp,power_dumped]=calculate_annual_supplied_load_and_penaltizing_objective_function(Nwt,Npv,N_batt,Pt,PPV_t,SE_batt,lpsplim,Power_demand)
    if (lpsp <= 0):

        return(0.96*(Ca/(asl/1000))+ 10**-11 * power_dumped)

    else :
        return((0.96*(Ca/(asl/1000)) +10**9* lpsp)+  10**-11 * power_dumped)#Objective function: Cost function to be minimized (simplified)




# Objective function: Cost function to be minimized (simplified)

# Simulated Annealing Algorithm
def simulated_annealing(initial_solution, temperature, cooling_rate,sa_fitness_track):

    #results = np.empty(227)

    current_solution = initial_solution
    best_solution = current_solution
    i=1
    best_fitness = objective_function(current_solution[0],current_solution[1],current_solution[2])


    while temperature > 0.1:

        current_fitness =objective_function(current_solution[0], current_solution[1],current_solution[2])
        new_solution = random_neighbor(current_solution,Nwt_min,Nwt_max,Npv_min,Npv_max,Nbatt_min,Nbatt_max)
        new_fitness=objective_function(new_solution[0], new_solution[1],new_solution[2])


        cost_diff = new_fitness - current_fitness
        r=random.random()
        p=math.exp(-cost_diff * 1.0 / temperature)

        if (cost_diff<0) or (cost_diff > 0 and (r < p)):
            current_solution = new_solution


        if current_fitness < best_fitness:
            best_solution = current_solution
            best_fitness=current_fitness
        sa_fitness_track = np.append(sa_fitness_track,current_fitness)


        temperature =initial_temperature - i*cooling_rate
        i=i+1
        print(i)





    return best_solution,sa_fitness_track
sa_solutions=np.loadtxt("sa_best_solutions")
sa_fitness_track=np.loadtxt("sa_fitness_track")
sa_best_fitness = np.loadtxt("sa_best_fitness")
for j in range (20) :
    initial_solution = [Nwt, Npv,N_batt]  # Initial guess for the number of wind turbines and PV panels

    initial_temperature = 10000
    cooling_rate = 100
    # Solve the problem using Simulated Annealing

    start_time=time.time()
    optimal_solution,sa_fitness_track = simulated_annealing(initial_solution, initial_temperature, cooling_rate,sa_fitness_track)
    end_time = time.time()
    print(start_time-end_time)
    print(f"the number of wind turbines is : {optimal_solution[0]}")
    print(f"the number of Pv panels is : {optimal_solution[1]}")
    print(f"the number of batteries is : {optimal_solution[2]}")
    print(objective_function(optimal_solution[0],optimal_solution[1],optimal_solution[2]))
    sa_best_fitness =np.append(sa_best_fitness,[objective_function(optimal_solution[0],optimal_solution[1],optimal_solution[2])])
    sa_solutions = np.concatenate((sa_solutions,optimal_solution),axis=0)



# print(sa_best_fitness)
# print(sa_fitness_track)
# print(sa_solutions)
np.savetxt("sa_best_fitness", sa_best_fitness, delimiter='\n')
np.savetxt("sa_best_solutions", sa_solutions ,delimiter='\n')
np.savetxt("sa_fitness_track",sa_fitness_track,delimiter='/n')














