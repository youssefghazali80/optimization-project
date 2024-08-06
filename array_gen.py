import random

import numpy as np

# Generate wind speed data as shown in the previous response
num_hours = 8760
mean_wind_speed = 5
std_dev = 3
wind_speed_data = np.random.normal(mean_wind_speed, std_dev, num_hours)
wind_speed_data = np.clip(wind_speed_data, 0, 25)

# Define a filename
filename = "wind_speed_data.txt"

# Save the data to a text file
np.savetxt(filename, wind_speed_data, delimiter='\n')
# Generate temperature data as shown in the previous response
num_hours = 8760
mean_temperature = 20.0
std_dev = 9
temperature_data = np.random.normal(mean_temperature, std_dev, num_hours)
temperature_data = np.clip(temperature_data, 10, 40)

# Define a filename for the temperature data
temperature_filename = "temperature_data.txt"

# Save the temperature data to a text file
np.savetxt(temperature_filename, temperature_data, delimiter='\n')

num_hours = 5110
mean_radiation = 1000
std_dev = 1500
radiation_data = np.random.normal(mean_radiation, std_dev, num_hours)
radiation_data = np.clip(radiation_data, 3000, 9000)
radiation_data= np.append(radiation_data,np.zeros(shape=(3650,1)))
random.shuffle(radiation_data)

# Define a filename for the solar radiation data
radiation_filename = "solar_radiation_data.txt"

# Save the solar radiation data to a text file
np.savetxt(radiation_filename, radiation_data, delimiter='\n')
# generate the power load demnand for each hour
np.random.seed(0)

# Specify the size of the array
array_size = 8760

# Specify the mean and standard deviation
mean_value = 35000
std_deviation = 15000

# Generate the array
power_demand_array = np.random.normal(mean_value, std_deviation, array_size)
power_demand_array=np.clip(power_demand_array,20000,60000)


# Define a filename for the power demand data
filename = "power_demand_data.txt"

# Save the power demand data to a text file
np.savetxt(filename, power_demand_array, delimiter='\n')




