import pulp
import time
import dimod
from dwave.system import DWaveSampler, EmbeddingComposite, LeapHybridSampler
import neal

# Costs of placing object i in box j
# For example, cost[i][j] represents the cost of placing object i in box j

# box - SV - V1, I12, I23
cost = [
    [300, None, None],   # Object 1 - V1
    [120, 120, None],    # Object 2 - V2
    [140, 140, 140],    # Object 3 - V3
    [None, 150, None],   # Object 4 - I1
    [None, 160, 160], # Object 5 - I2 
    [None, None, 150], # Object 6 - I3
    [None, 300, None], # Object 7 - I12
    [None, None, 300]   # Object 8 - I23
]

# Define objects, boxes, and costs
num_objects = len(cost)
num_boxes = len(cost[0])


# Define the problem
problem = pulp.LpProblem("Object_Assignment", pulp.LpMinimize)

# Define decision variables
# x[i][j] is 1 if object i is placed in box j, and 0 otherwise
x = [[pulp.LpVariable(f"x_{i}_{j}", cat="Binary") for j in range(num_boxes)] for i in range(num_objects)]

# Objective function: minimize the cost
problem += pulp.lpSum(cost[i][j] * x[i][j] for i in range(num_objects) for j in range(num_boxes) if cost[i][j] is not None)

# Constraint 1: Each object is assigned to exactly one box
for i in range(num_objects):
    problem += pulp.lpSum(x[i][j] for j in range(num_boxes) if cost[i][j] is not None) <= 1, f"Object_{i}_assigned_once"

# Constraint 2: Each box has exactly one object
for j in range(num_boxes):
    problem += pulp.lpSum(x[i][j] for i in range(num_objects) if cost[i][j] is not None) == 1, f"Box_{j}_has_one_object"

# Solve the problem using a classical solver (e.g., CBC, Gurobi, etc.)
solver = pulp.PULP_CBC_CMD()  # You can use other solvers like Gurobi or CPLEX if available
# Measure time for classical solver (PuLP)
start_time_classical = time.time()
problem.solve(solver)
end_time_classical = time.time()
classical_time = end_time_classical - start_time_classical

# Display the results
print("Classical Solution")
print("Status :", pulp.LpStatus[problem.status])
print("Total Cost :", pulp.value(problem.objective))

# Print object assignments
for i in range(num_objects):
    for j in range(num_boxes):
        if pulp.value(x[i][j]) == 1:
            print(f"Classical: Object {i+1} is placed in Box {j+1}")


############################### QUANTUM #########################

# Define the binary variables x[i][j] for object i and box j
linear = {}
quadratic = {}
# Define penalty multipliers
lambda_object = 600 # Penalize placing an object in more than one box
lambda_box = 600    # Penalize placing more than one object in a box

#### new quantum ####
num_objects = len(cost)
num_boxes = len(cost[0])

# Create a Binary Quadratic Model (BQM)
bqm = dimod.BinaryQuadraticModel('BINARY')

# Define decision variables: x[i][j] is 1 if object i is assigned to box j
x = [[f'x_{i}_{j}' for j in range(num_boxes)] for i in range(num_objects)]

# Objective: Minimize the total cost
for i in range(num_objects):
    for j in range(num_boxes):
        if cost[i][j] is not None:
            bqm.add_variable(x[i][j], cost[i][j])

# New constraint: Each object can be placed in at most one box
for i in range(num_objects):
    # Sum of all boxes for a given object must be <= 1 (at most one box)
    bqm.add_linear_inequality_constraint(
        [(x[i][j], 1) for j in range(num_boxes) if cost[i][j] is not None],
        lb=0, ub=1,  # Lower bound 0 (can be unassigned), upper bound 1 (at most one box)
        lagrange_multiplier=lambda_object,
        label=f"object_{i}_assignment"
    )

# Constraint: Each box must contain exactly one object
for j in range(num_boxes):
    # Sum of all objects assigned to a given box must be exactly 1
    bqm.add_linear_equality_constraint(
        [(x[i][j], 1) for i in range(num_objects) if cost[i][j] is not None],
        constant=-1,
        lagrange_multiplier=lambda_box
    )

# number of reads 
n_reads = 100


# simulated aneealer
#sampler = dimod.SimulatedAnnealingSampler()
sampler = neal.sampler.SimulatedAnnealingSampler()

# Solve the BQM using the D-Wave Hybrid Sampler
sampler_hybrid = LeapHybridSampler()

# Solve the problem using a D-Wave sampler
sampler_qpu = EmbeddingComposite(DWaveSampler())


start_sim = time.time()
sampleset = sampler.sample(bqm, num_reads=n_reads) 
end_sim = time.time()

start_hybrid = time.time()
sampleset_hybrid = sampler_hybrid.sample(bqm) 
end_hybrid = time.time()

start_qpu = time.time()
sampleset_qpu = sampler_qpu.sample(bqm) 
end_qpu = time.time()

# Output total time taken:
print("Time taken by simulated annealer: ", start_sim - end_sim)
print("Time taken by hybrid solver: ", start_hybrid - end_hybrid)
print("Time taken by QPU solver: ", start_qpu - end_qpu)

# Output the results
print("Simulated Annealer Solution Objective value:", sampleset.first.energy)
#print("Hybrid Solution Objective value:", sampleset_hybrid.first.energy)
#print("QPU Solution Objective value:", sampleset_qpu.first.energy)


# Get the best solution
best_solution = sampleset.first.sample
best_solution_hybrid = sampleset_hybrid.first.sample
best_solution_qpu = sampleset_qpu.first.sample

for i in range(num_objects):
    for j in range(num_boxes):
        var_name = f'x_{i}_{j}'
        if best_solution.get(var_name) == 1:
            print(f"Simulated Annealer: Object {i + 1} is placed in Box {j + 1}")
        if best_solution_hybrid.get(var_name) == 1:
            print(f"Hybrid Solver: Object {i + 1} is placed in Box {j + 1}")
        if best_solution_qpu.get(var_name) == 1:
            print(f"QPU: Object {i + 1} is placed in Box {j + 1}")



print("All combinations with similar energy (with QPU):")
option = 1
for s in sampleset_qpu.data():
    if s.energy == sampleset_qpu.first.energy:
        print(s)
        print(f"---- Option {option}-----")
        option += 1
        for i in range(num_objects):
            for j in range(num_boxes):
                var_name = f'x_{i}_{j}'
                if s.sample.get(var_name) == 1:
                    print(f"Quantum: Object {i + 1} is placed in Box {j + 1}")
