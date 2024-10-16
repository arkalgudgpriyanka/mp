import dimod
import time
import neal

# Number of objects and boxes
num_objects = 8
num_boxes = 3

# Costs of objects for each box (None means the object cannot be assigned to that box)
# box - SV - V1, I12, I23
costs = [
    [300, None, None],   # Object 1 - V1
    [120, 120, None],    # Object 2 - V2
    [140, 140, 140],    # Object 3 - V3
    [None, 150, None],   # Object 4 - I1
    [None, 160, 160], # Object 5 - I2 
    [None, None, 150], # Object 6 - I3
    [None, 300, None], # Object 7 - I12
    [None, None, 300]   # Object 8 - I23
]

# Define profits for objects in each box (same format as costs)
profits = [
    [10, None, None],   # Object 1 - V1
    [6, 6, None],    # Object 2 - V2
    [4, 4, 4],    # Object 3 - V3
    [None, 8, None],   # Object 4 - I1
    [None, 8, 8], # Object 5 - I2 
    [None, None, 8], # Object 6 - I3
    [None, 10, None], # Object 7 - I12
    [None, None, 10]   # Object 8 - I23
]

# Global budget for all boxes combined
global_budget = 500  # Example budget

############################### QUANTUM #########################



# Define penalty multipliers
lambda_object = 600 # Penalize placing an object in more than one box
lambda_box = 600    # Penalize placing zero object in a box
lambda_budget = 600 # Penalize total cost of all objects placed in all boxes more than the global budget

#### new quantum ####
num_objects = len(costs)
num_boxes = len(costs[0])

# Create a Binary Quadratic Model (BQM)
bqm = dimod.BinaryQuadraticModel('BINARY')

# Define decision variables: x[i][j] is 1 if object i is assigned to box j
x = [[f'x_{i}_{j}' for j in range(num_boxes)] for i in range(num_objects)]

# Objective: Maximize the total profit
for i in range(num_objects):
    for j in range(num_boxes):
        if profits[i][j] is not None:
            bqm.add_variable(x[i][j], -profits[i][j])

# New constraint: Each object can be placed in at most one box
for i in range(num_objects):
    # Sum of all boxes for a given object must be <= 1 (at most one box)
    bqm.add_linear_inequality_constraint(
        [(x[i][j], 1) for j in range(num_boxes) if costs[i][j] is not None],
        lb=0, ub=1,  # Lower bound 0 (can be unassigned), upper bound 1 (at most one box)
        lagrange_multiplier=lambda_object,
        label=f"object_{i}_assignment"
    )

# Constraint 2: Each box must contain atleast one object
for j in range(num_boxes):
    # Sum of all objects for a given box must be >= 1 and <=8 (at least one object per box)
    bqm.add_linear_inequality_constraint(
        [(x[i][j], 1) for i in range(num_objects) if costs[i][j] is not None],
        lb=1, ub=8,  # Lower bound 1 (atleast 1 object assigned), upper bound 8 (at most 8 objects assigned)
        lagrange_multiplier=lambda_object,
        label=f"box_{j}_assignment"
    )

# Constraint 3 : Total cost of all objects in all boxes must be within the global budget
#cost_constraints = [((i * num_boxes + j), costs[i][j]) for i in range(num_objects) for j in range(num_boxes) if costs[i][j] is not None]
cost_constraints = [(x[i][j], costs[i][j]) for i in range(num_objects) for j in range(num_boxes) if costs[i][j] is not None]
bqm.add_linear_inequality_constraint(cost_constraints, 
                                     lb=0, ub=global_budget,  # Lower bound 0 (total cost is 0), upper bound is global budget
                                    lagrange_multiplier=lambda_budget,
                                    label='total_cost_limit')

# simulated aneealer
#sampler = dimod.SimulatedAnnealingSampler()
sampler = neal.sampler.SimulatedAnnealingSampler()
n_reads = 200
start_time_simannealer = time.time()
sampleset = sampler.sample(bqm, num_reads=n_reads)
end_time_simannealer = time.time()
time_simannealer = end_time_simannealer - start_time_simannealer
print(f"Total time taken for {n_reads} reads is {time_simannealer}")

print("size of total sampleset: ", len(sampleset))
# Output the results
print("Quantum Solution Objective value:", -sampleset.first.energy)
# Get the best solution
best_solution = sampleset.first.sample

for i in range(num_objects):
    for j in range(num_boxes):
        var_name = f'x_{i}_{j}'
        if best_solution.get(var_name) == 1:
            print(f"Quantum: Object {i + 1} is placed in Box {j + 1}")

print("All combinations with similar energy:")
option = 1
for s in sampleset.data():
    if s.energy == sampleset.first.energy:
        print(f"---- Quantum Option {option}-----")
        total_cost = 0
        total_profit = 0
        option += 1
        for i in range(num_objects):
            for j in range(num_boxes):
                var_name = f'x_{i}_{j}'
                if s.sample.get(var_name) == 1:
                    print(f"Object {i + 1} is placed in Box {j + 1}")
                    total_cost = total_cost + costs[i][j]
                    total_profit = total_profit + profits[i][j]
        print("Energy: ", -s.energy)
        print("Total costs incurred: ", total_cost)
        print("Total profits incurred: ", total_profit)


#print("------ All samplesets --------")
#print(sampleset)


