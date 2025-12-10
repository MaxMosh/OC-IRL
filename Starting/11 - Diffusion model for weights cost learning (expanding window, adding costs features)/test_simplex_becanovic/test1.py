from simplex_grid import simplex_grid

# Generate points on a 2D simplex (triangle)
grid = simplex_grid(m=2, r=5)
print(f"Generated {len(grid)} points")

# Visualize the points
import matplotlib.pyplot as plt
points = [p.flatten() for p in grid]
plt.scatter([p[0] for p in points], [p[1] for p in points])
plt.xlabel('w_0')
plt.ylabel('w_1')
plt.title('Points on 2D Simplex')
plt.show()