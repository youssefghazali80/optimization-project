import random
import numpy as np
import time
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
Nwt_min = 1
Nwt_max = 40
Npv_min = 100
Npv_max = 1000
Nbatt_min = int(calculate_the_minimun_number_batteries(Eload,nd,nb,Dod,vb,Cb))
Nbatt_max = 500
Nwt = random.randint(1,20)
Npv = random.randint(100,1000)
N_batt  = random.randint(Nbatt_min,500)
lpsplim = 0.2 # loss of power supply index
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

def check_feasibility(current_solution, Nwt_min, Nwt_max, Npv_min, Npv_Max, Nbatt_Min, Nbatt_Max):
    # Generate a neighboring solution by perturbing the current solution
    wind_turbines, pv_panels, batteries = current_solution

    # Ensure that the number of turbines, panels, and batteries is within the boudaries
    new_wind_turbines = max(wind_turbines, Nwt_min)
    new_wind_turbines = min(new_wind_turbines, Nwt_max)
    new_pv_panels = min(pv_panels, Npv_Max)
    new_pv_panels = max(new_pv_panels, Npv_min)
    new_batteries = max(batteries, Nbatt_Min)
    new_batteries = min(new_batteries, Nbatt_Max)
    return [new_wind_turbines, new_pv_panels, new_batteries]
def teaching_phase(Nwt_min ,Nwt_max,Npv_min,Npv_Max ,Nbatt_Min,Nbatt_Max,pop_size,current_solution,fitness_value,mean):
    new_solution= np.empty(shape=current_solution.shape)
    teacher = current_solution[np.argpartition(fitness_value.flatten(), 1)[:1]]
    new_fitness_value = np.empty(shape=(pop_size * 1))
    for i in range (pop_size):
        new_solution[i] = current_solution[i] + ( random.random() *(teacher-random.randint(1,2)*mean))
    new_solution = np.round(new_solution)
    for z in range (pop_size):
        new_solution[z] = check_feasibility(new_solution[z],Nwt_min ,Nwt_max,Npv_min,Npv_Max ,Nbatt_Min,Nbatt_Max)
    for j in range(pop_size):
        new_fitness_value[j] = objective_function(new_solution[j,0], new_solution[j,1], new_solution[j,2])
    for k in range (pop_size):
        if (new_fitness_value[k]<fitness_value[k]):
            current_solution[k] = new_solution[k]
            fitness_value[k] =new_fitness_value[k]

    return current_solution , fitness_value
def learning_phase(Nwt_min ,Nwt_max,Npv_min,Npv_Max ,Nbatt_Min,Nbatt_Max,pop_size,current_solution,fitness_value):
    new_solution = np.empty(shape=current_solution.shape)
    new_fitness_value = np.empty(shape=(pop_size * 1))
    for i in range (pop_size):
        r=random.randint(0,pop_size-1)
        if (fitness_value[i]>fitness_value[r]):
            new_solution[i] = current_solution[i]+(random.random()*(current_solution[r]-current_solution[i]))
        else:
            new_solution[i] = current_solution[i] + (random.random() * (current_solution[i] - current_solution[r]))
        new_solution = np.round(new_solution)
        for z in range (pop_size):

            new_solution[z] = check_feasibility(new_solution[z], Nwt_min, Nwt_max, Npv_min, Npv_Max, Nbatt_Min, Nbatt_Max)
        for j in range(pop_size):
            new_fitness_value[j] = objective_function(new_solution[j, 0], new_solution[j, 1], new_solution[j, 2])
        for k in range(pop_size):
            if (new_fitness_value[k] < fitness_value[k]):     #updating the solutions if the new solutiona are better
                current_solution[k] = new_solution[k]
                fitness_value[k] = new_fitness_value[k]

    return  current_solution,fitness_value






TLBO_best_solutions = np.loadtxt("TLBO_best_solutions")
TLBO_best_fitness = np.loadtxt("TLBO_best_fitness")
TLBO_fitness_track = np.loadtxt("TLBO_fitness_track")

def TLBO_Algorithm(Nwt_min ,Nwt_max,Npv_min,Npv_Max ,Nbatt_Min,Nbatt_Max,pop_size,num_of_iterations,TLBO_fitness_track):
    current_solution = generate_initial_solutions(Nwt_min, Nwt_max, Npv_min, Npv_Max, Nbatt_Min, Nbatt_Max, pop_size)#initializng an initial learners(solutions)
    fitness_value = np.empty(shape=(pop_size * 1))
    for i in range(pop_size):
        fitness_value[i] = objective_function(current_solution[i, 0], current_solution[i, 1], current_solution[i, 2])
    best_solution =  current_solution[np.argpartition(fitness_value.flatten(), 1)[:1]]
    best_fitness = fitness_value[np.argpartition(fitness_value.flatten(), 1)[:1]]
    for k in range(num_of_iterations):
        TLBO_fitness_track = np.append(TLBO_fitness_track, [best_fitness])
        print(k)
        #for i in range (pop_size):
            #fitness_value[i]=objective_function(current_solution[i,0],current_solution[i,1],current_solution[i,2])
        mean = np.mean(current_solution,axis=0)
        current_solution , fitness_value = teaching_phase(Nwt_min ,Nwt_max,Npv_min,Npv_Max ,Nbatt_Min,Nbatt_Max,pop_size,current_solution,fitness_value,mean) # applying the teaching phase on our learners and updating the solution
        current_solution ,fitness_value = learning_phase(Nwt_min ,Nwt_max,Npv_min,Npv_Max ,Nbatt_Min,Nbatt_Max,pop_size,current_solution,fitness_value)   # applying the learning phase on our learners
        new_best_fitness =  fitness_value[np.argpartition(fitness_value.flatten(), 1)[:1]]
        if(new_best_fitness<best_fitness):
            best_solution=current_solution[np.argpartition(fitness_value.flatten(), 1)[:1]]
            best_fitness=fitness_value[np.argpartition(fitness_value.flatten(), 1)[:1]]



    return best_solution,best_fitness,TLBO_fitness_track










for k in range(1):
    start_time = time.time()
    best_solution, best_fitness, TLBO_fitness_track = TLBO_Algorithm(Nwt_min, Nwt_max, Npv_min, Npv_max, Nbatt_min,Nbatt_max, 30, 100,TLBO_fitness_track)
    end_time =time.time()
    print(end_time-start_time)
    print(f"the best solution is :{best_solution[0]}, /n the objective function is {best_fitness}")
    TLBO_best_fitness = np.append(TLBO_best_fitness,[best_fitness])
    TLBO_best_solutions = np.concatenate((TLBO_best_solutions,best_solution[0]),axis=0)
    print(k)

#np.savetxt("TLBO_best_fitness", TLBO_best_fitness, delimiter='\n')
#np.savetxt("TLBO_best_solutions", TLBO_best_solutions ,delimiter='\n')
#np.savetxt("TLBO_fitness_track",TLBO_fitness_track,delimiter='/n')