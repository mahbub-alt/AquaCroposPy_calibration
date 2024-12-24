from aquacrop import AquaCropModel, Soil, Crop, InitialWaterContent,IrrigationManagement
from aquacrop.utils import prepare_weather, get_filepath
import scipy.stats
import numpy as np
print("strt")
import Aq_Optimization_functons
from Aq_Optimization_functons import *
import warnings
from pyswarm import pso
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import ThreadPoolExecutor
warnings.filterwarnings("ignore")
print ("imported")
import pandas as pd
import time

# Record the start time
start_time = time.time()
                        # CCx = 0.96, WP = 33.7,  Kcb =1.05, HI0 = 0.48, a_HI = 7.0, Zmax = 2.3 defaults

lb = [30, 50,   50,   30,    .85,  30, 1,   .45, .5,  1.20]  # Lower bound
ub = [60, 65,   65,   60,    .98,  35, 1.10, .55, 10,  2.5]  # Upper bound
minfunc = 1e-2
swarmsize = 50
maxiter = 100
print("swrm:",  swarmsize)                                                 
print ('iter:', maxiter)

param_names = ["SMT1", "SMT2", "SMT3", "SMT4", "CCx", "WP", "Kcb", "HI0", "a_HI", "Zmax"]

# results = []

# # Run PSO optimization 3 times
# for i in range(3):
#     # print(f"Running PSO optimization {i+1}/10")
#     xopt, fopt = pso(obj_func, lb, ub, swarmsize=swarmsize, maxiter=maxiter, args =(False,True))
#     result = [i+1] + list(xopt) + [fopt]  # Include the iteration number
#     results.append(result)

# # Create a DataFrame with the optimized parameters
# df = pd.DataFrame(results, columns=['Iteration'] + param_names + ['Fitness'])

# # Export the DataFrame to a CSV file
# # df.to_csv('optimized_parameters_multiple_runs.csv', index=False)
# print (df)

      # testing part.------------------------------  # no_ET =True, Train = True

def run_optimization(iteration):
    np.random.seed(iteration)
    xopt, fopt = pso(obj_func, lb, ub, swarmsize=swarmsize, maxiter=maxiter, args=(False, False))
    return [iteration] + list(xopt) + [fopt]

# Use ProcessPoolExecutor for parallel execution
num_parallel_runs = 30  # Change this to the number of CPUs you want to use
with ProcessPoolExecutor(max_workers=num_parallel_runs) as executor:
    results = list(executor.map(run_optimization, range(1, num_parallel_runs + 1)))
# Create a DataFrame to store results
df = pd.DataFrame(results, columns=['Iteration'] + param_names + ['Fitness'])

# Export the DataFrame to a CSV file
df.to_csv('50particle_100_withET_newobj_fulldata.csv', index=False)
print(df)
end_time = time.time()

# Calculate the time taken
elapsed_time = (end_time - start_time)/3600
print(f"Elapsed time: {elapsed_time:.2f} hours")