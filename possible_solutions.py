import itertools

# Define profits, costs, and affiliations (where None means that the object cannot go into that box)
costs = [
    [10, 20, 15],   # Object 1
    [25, 30, 5],    # Object 2
    [8, 22, 17],    # Object 3
    [30, 18, 10],   # Object 4
    [12, None, 16], # Object 5
    [None, 14, 22], # Object 6
    [20, None, 18], # Object 7
    [9, 21, None]   # Object 8
]

# Define profits for objects in each box (same format as costs)
profits = [
    [10, 15, 12],   # Profit for Object 1
    [18, 20, 7],    # Profit for Object 2
    [6, 18, 12],    # Profit for Object 3
    [25, 10, 8],    # Profit for Object 4
    [8, None, 14],  # Profit for Object 5
    [None, 9, 18],  # Profit for Object 6
    [15, None, 14], # Profit for Object 7
    [7, 16, None]   # Profit for Object 8
]
# Predefined total cost range for all objects placed in boxes combined
total_cost_min = 0
total_cost_max = 100

num_objects = len(profits)
num_boxes = len(profits[0])

# Generate all possible assignments
all_possible_assignments = list(itertools.product(range(num_boxes), repeat=num_objects))

# Function to check if an assignment is feasible
def is_feasible(assignment):
    box_occupancy = [0] * num_boxes
    total_cost = 0
    
    for obj in range(num_objects):
        box = assignment[obj]
        if profits[obj][box] is None:  # Check affiliation
            return False
        box_occupancy[box] += 1  # Mark that this box has one object
        total_cost += costs[obj][box] if costs[obj][box] is not None else 0

    # Check the total cost and that each box has at least one object
    if total_cost_min <= total_cost <= total_cost_max and all(occupancy > 0 for occupancy in box_occupancy):
        return True
    return False

# Find all feasible solutions
feasible_solutions = []
for assignment in all_possible_assignments:
    if is_feasible(assignment):
        feasible_solutions.append(assignment)

# Print feasible solutions
print("Feasible Solutions (Object assignments to boxes):")
for solution in feasible_solutions:
    print(solution)
