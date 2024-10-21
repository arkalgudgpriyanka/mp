import dimod
import time
import neal
import pulp
import dwave.inspector
from dwave.system import DWaveSampler, EmbeddingComposite, LeapHybridSampler

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


##################### CLASSICAL ######################################
# Create the LP problem
prob = pulp.LpProblem("Box_Object_Optimization", pulp.LpMaximize)

# Define decision variables
x = pulp.LpVariable.dicts("x", (range(len(profits)), range(len(profits[0]))), cat='Binary')

# Objective function: Maximize total profit
prob += pulp.lpSum(profits[i][j] * x[i][j] for i in range(len(profits)) for j in range(len(profits[0])) if profits[i][j] is not None), "Total_Profit"

# Constraint: Each object can be placed in at most one box
for i in range(len(profits)):
    prob += pulp.lpSum(x[i][j] for j in range(len(profits[0])) if profits[i][j] is not None) <= 1, f"One_Box_per_Object_{i}"

# Constraint: Each box must contain at least one object
for j in range(len(profits[0])):
    prob += pulp.lpSum(x[i][j] for i in range(len(profits)) if profits[i][j] is not None) >= 1, f"AtLeast_One_Object_in_Box_{j}"

# Constraint: Total cost of selected objects must be less than or equal to the limit
prob += pulp.lpSum(costs[i][j] * x[i][j] for i in range(len(profits)) for j in range(len(profits[0])) if costs[i][j] is not None) <= global_budget, "Total_Cost_Limit"

# Solve the problem
start_time_classical = time.time()
prob.solve()
end_time_classical = time.time()
classical_time = end_time_classical - start_time_classical

# Output results
print("Status:", pulp.LpStatus[prob.status])
print("Optimal Assignment:")
for i in range(len(profits)):
    for j in range(len(profits[0])):
        if pulp.value(x[i][j]) == 1:
            print(f"Object {i + 1} assigned to Box {j + 1}")

total_profit = pulp.value(prob.objective)
total_cost = sum(costs[i][j] * pulp.value(x[i][j]) for i in range(len(profits)) for j in range(len(profits[0])) if costs[i][j] is not None)

print("---------Classical Solutions---------")
print(f"\nTotal Profit: {total_profit}")
print(f"Total Cost: {total_cost}")
print("Total time taken: ", classical_time)


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


# number of reads
n_reads = 5000

# simulated aneealer
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
sampleset_qpu = sampler_qpu.sample(bqm, num_reads=n_reads) 
end_qpu = time.time()

# Output total time taken:
print("Time taken by simulated annealer: ", end_sim - start_sim)
print("Time taken by hybrid solver: ", end_hybrid - start_hybrid)
print("Time taken by QPU solver: ", end_qpu - start_qpu)

# Output the results
print("Simulated Annealer Solution Objective value:", -sampleset.first.energy)
print("Hybrid Solver Solution Objective value:", -sampleset_hybrid.first.energy)
print("QPU Solution Objective value:", -sampleset_qpu.first.energy)

print("\sampleset_qpu:")
print(sampleset_qpu.first)
#print(sampleset_qpu)
# open inspector
dwave.inspector.show(sampleset_qpu)

# Get the best solution
best_solution = sampleset.first.sample
best_solution_hybrid = sampleset_hybrid.first.sample
best_solution_qpu = sampleset_qpu.first.sample

print("---------- Simulated Annealer Best solution----------------")
for i in range(num_objects):
    for j in range(num_boxes):
        var_name = f'x_{i}_{j}'
        if best_solution.get(var_name) == 1:
            print(f"Object {i + 1} is placed in Box {j + 1}")

print("---------- Hybrid Solver Best solution----------------")
for i in range(num_objects):
    for j in range(num_boxes):
        var_name = f'x_{i}_{j}'
        if best_solution_hybrid.get(var_name) == 1:
            print(f"Object {i + 1} is placed in Box {j + 1}")

print("---------- QPU Best solution----------------")
for i in range(num_objects):
    for j in range(num_boxes):
        var_name = f'x_{i}_{j}'
        if best_solution_qpu.get(var_name) == 1:
            print(f"Object {i + 1} is placed in Box {j + 1}")

embedding = sampleset_qpu.info['embedding_context']['embedding']
print(f"Number of logical variables: {len(embedding.keys())}")
print(f"Number of physical qubits used in embedding: {sum(len(chain) for chain in embedding.values())}")

print("All combinations with similar energy (with QPU):")
option = 1
for s in sampleset_qpu.data():
    if s.energy == sampleset_qpu.first.energy:
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

print("total options with similar energy: ", option-1)


print("Print first 5 QPU samples")

res = 1
for s in sampleset_qpu.data():
        print(f"---- Quantum result {res}-----")
        total_cost = 0
        total_profit = 0
        res += 1
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
        if res == 6:
            break
