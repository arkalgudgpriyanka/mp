import pulp

# Define profits and costs
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

# Predefined total cost limit
total_cost_limit = 500

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
prob += pulp.lpSum(costs[i][j] * x[i][j] for i in range(len(profits)) for j in range(len(profits[0])) if costs[i][j] is not None) <= total_cost_limit, "Total_Cost_Limit"

# Solve the problem
prob.solve()

# Output results
print("Status:", pulp.LpStatus[prob.status])
print("Optimal Assignment:")
for i in range(len(profits)):
    for j in range(len(profits[0])):
        if pulp.value(x[i][j]) == 1:
            print(f"Object {i + 1} assigned to Box {j + 1}")

total_profit = pulp.value(prob.objective)
total_cost = sum(costs[i][j] * pulp.value(x[i][j]) for i in range(len(profits)) for j in range(len(profits[0])) if costs[i][j] is not None)

print(f"\nTotal Profit: {total_profit}")
print(f"Total Cost: {total_cost}")
