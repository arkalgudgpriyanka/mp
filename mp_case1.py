import pulp
import time
import dimod
from dwave.system import DWaveSampler, EmbeddingComposite, LeapHybridSampler

# Costs of placing object i in box j
# For example, cost[i][j] represents the cost of placing object i in box j

cost = [
    [10, 20, 15],  # Object 1 can be placed in box 1, 2, or 3 with respective costs
    [25, 30, 5],   # Object 2 can be placed in box 1, 2, or 3 with respective costs
    [8, 22, 17],   # Object 3 can be placed in box 1, 2, or 3 with respective costs
    [30, 18, 10],  # Object 4 can be placed in box 1, 2, or 3 with respective costs
    [12, None, 16], # Object 5 can be placed in box 1 or 3 but not in box 2
    [None, 14, 22], # Object 6 can be placed in box 2 or 3 but not in box 1
    [20, None, 18], # Object 7 can be placed in box 1 or 3 but not in box 2
    [9, 21, None]  # Object 8 can be placed in box 1 or 2 but not in box 3
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
lambda_object = 20 # Penalize placing an object in more than one box
lambda_box = 20     # Penalize placing more than one object in a box

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

# Solve the BQM using the D-Wave Hybrid Sampler
sampler_hybrid = LeapHybridSampler()
# Solve the problem using a D-Wave sampler
sampler = EmbeddingComposite(DWaveSampler())

# Run D-Wave quantum solver and measure time
start_time_quantum = time.time()
sampleset = sampler.sample(bqm, num_reads=100)
end_time_quantum = time.time()
quantum_time = end_time_quantum - start_time_quantum

# Run D-Wave quantum solver and measure time
start_time_quantum_hybrid = time.time()
sampleset_hybrid = sampler_hybrid.sample(bqm, num_reads=100)
end_time_quantum_hybrid = time.time()
quantum_time_hybrid = end_time_quantum_hybrid - start_time_quantum_hybrid

# Get the best solution
best_solution = sampleset.first.sample

# Manual feasibility check
feasible = True

# Check object assignment constraints (each object must be assigned to at most one box)
for i in range(num_objects):
    assigned_boxes = sum(best_solution[f'x_{i}_{j}'] for j in range(num_boxes) if cost[i][j] is not None)
    if assigned_boxes > 1:
        feasible = False
        print(f"Object {i + 1} is assigned to {assigned_boxes} boxes, which violates the constraints.")

# Check box assignment constraints (each box must contain exactly one object)
for j in range(num_boxes):
    assigned_objects = sum(best_solution[f'x_{i}_{j}'] for i in range(num_objects) if cost[i][j] is not None)
    if assigned_objects != 1:
        feasible = False
        print(f"Box {j + 1} has {assigned_objects} objects, which violates the constraints.")

if feasible:
    print("Quantum Found a feasible solution!")
else:
    print("Quantum No feasible solution found.")

# Output the results
print("Objective value:", sampleset.first.energy)
for i in range(num_objects):
    for j in range(num_boxes):
        var_name = f'x_{i}_{j}'
        if best_solution.get(var_name) == 1:
            print(f"Object {i + 1} is placed in Box {j + 1}")


##################### Comparison of execution time of Classical and Quantum ################
# Compare execution times
print(f"Classical solver execution time: {classical_time} seconds")
print(f"Quantum solver execution time: {quantum_time} seconds")
print(f"Quantum hybrid solver execution time: {quantum_time_hybrid} seconds")

# Compare results (cost)
classical_cost = pulp.value(problem.objective)

# Initialize quantum_cost
quantum_cost = sampleset.first.energy

print(f"Classical solver cost: {classical_cost}")
print(f"Quantum solver cost: {quantum_cost}")