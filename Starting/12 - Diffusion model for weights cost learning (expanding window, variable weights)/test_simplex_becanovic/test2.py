from simplex_grid import simplex_grid
import numpy as np

# Generate weight vectors for 3-objective optimization
m = 3  # number of objectives
r = 11  # resolution (higher = more points)

weights = simplex_grid(m=m, r=r)
print(f"Generated {len(weights)} weight vectors")

# Use in multi-objective optimization
print(type(weights))
for w in weights:
    # Each w is a weight vector with shape (3, 1)
    # Use it to combine multiple objectives
    # combined_objective = w[0] * obj1 + w[1] * obj2 + w[2] * obj3
    # print(w)
    # print(w.sum())
    print(type(w))
    pass