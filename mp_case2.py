import dimod
from dwave.system import LeapHybridSampler

# Example Data
profits = [
    [10, 20, 15],  # Profit for object 1 in box 1, 2, 3
    [25, 30, 5],   # Profit for object 2 in box 1, 2, 3
    [8, 22, 17],   # Profit for object 3 in box 1, 2, 3
    [30, 18, 10],  # Profit for object 4 in box 1, 2, 3
    [12, None, 16],# Profit for object 5 in box 1 or 3 (None means no affiliation)
    [None, 14, 22],# Profit for object 6 in box 2 or 3 (None means no affiliation)
    [20, None, 18],# Profit for object 7 in box 1 or 3 (None means no affiliation)
    [9, 21, None]  # Profit for object 8 in box 1 or 2 (None means no affiliation)
]

costs = [
    [10, 20, 15],  # Cost for object 1 in box 1, 2, 3
    [25, 30, 5],   # Cost for object 2 in box 1, 2, 3
    [8, 22, 17],   # Cost for object 3 in box 1, 2, 3
    [30, 18, 10],  # Cost for object 4 in box 1, 2, 3
    [12, None, 16],# Cost for object 5 in box 1 or 3 (None means no affiliation)
    [None, 14, 22],# Cost for object 6 in box 2 or 3 (None means no affiliation)
    [20, None, 18],# Cost for object 7 in box 1 or 3 (None means no affiliation)
    [9, 21, None]  # Cost for object 8 in box 1 or 2 (None means no affiliation)
]

# Predefined total cost range for all objects placed in boxes combined
total_cost_min = 140  # Example minimum total cost
total_cost_max = 160  # Example maximum total cost

num_objects = len(profits)
num_boxes = len(profits[0])

# Create a Binary Quadratic Model (BQM)
bqm = dimod.BinaryQuadraticModel('BINARY')

# Define decision variables: x[i][j] is 1 if object i is assigned to box j
x = [[f'x_{i}_{j}' for j in range(num_boxes)] for i in range(num_objects)]

# Objective: Maximize the total profit
for i in range(num_objects):
    for j in range(num_boxes):
        if profits[i][j] is not None:  # Only consider valid box affiliations
            bqm.add_variable(x[i][j], -profits[i][j])  # Maximize profit (minimize negative profit)

# Constraint 1: Each object can be placed in at most one box
for i in range(num_objects):
    bqm.add_linear_inequality_constraint(
        [(x[i][j], 1) for j in range(num_boxes) if profits[i][j] is not None],
        lb=0, ub=1,  # At most one box per object
        lagrange_multiplier=10,  # Adjust multiplier based on problem size
        label=f"object_{i}_placement"
    )

# Relaxed Constraint 2: Each box must contain at least one object
for j in range(num_boxes):
    valid_terms = [(x[i][j], 1) for i in range(num_objects) if profits[i][j] is not None]
    
    # Allow soft constraint: Encourage at least one object per box, but don't strictly enforce it
    if valid_terms:  # Only if there are valid objects for the box
        bqm.add_linear_inequality_constraint(
            terms=valid_terms,
            lb=0,  # Allow empty boxes but penalize them
            lagrange_multiplier=5,  # Softer constraint than the hard placement constraint
            label=f"box_{j}_soft_capacity"
        )

# Constraint 3: Total cost of all objects placed in boxes must be within a range [total_cost_min, total_cost_max]
valid_terms = [(x[i][j], costs[i][j]) for i in range(num_objects) for j in range(num_boxes) if costs[i][j] is not None]

# Lower bound constraint: Total cost must be >= total_cost_min
bqm.add_linear_inequality_constraint(
    terms=valid_terms,
    lb=total_cost_min,  # Minimum total cost
    ub=total_cost_max,  # Maximum total cost
    lagrange_multiplier=10,
    label="total_combined_cost_range"
)

# Solve the BQM using the D-Wave Hybrid Sampler
sampler = LeapHybridSampler()
result = sampler.sample(bqm)

# Get the best solution
best_solution = result.first.sample

# Output the results
print("Objective value:", -result.first.energy)  # Maximize profit (energy is negative of profit)
for i in range(num_objects):
    for j in range(num_boxes):
        if best_solution.get(f'x_{i}_{j}') == 1:
            print(f"Object {i + 1} is placed in Box {j + 1}")

# Check if the solution is feasible
if result.first.is_feasible:
    print("Feasible solution found!")
else:
    print("No feasible solution found.")
