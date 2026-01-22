# Particle-Swarm-Optimization
Improved Particle Swarm Optimization with 2D swarm visualization 
 Features

- **Iteractive user input**: Choose function, dimension (best at 2D), particles, iterations — no code editing needed
- **Linear inertia weight decay**: From w_max=0.9 to w_min=0.39 for balanced exploration → exploitation
- **Velocity clamping** + simple reflection boundary handling
- **Focused visualization**: Clean final swarm scatter plot on function contour (no convergence graph or animation clutter)
- **Benchmark functions included**:
  - Sphere (unimodal)
  - Rastrigin (multimodal, many local minima)
  - Rosenbrock (valley-shaped, non-convex)
- **Beautiful contour background** using matplotlib (magma colormap + logarithmic levels when needed)
- **Global best highlighted** with large lime star marker

## Why this is improved over basic PSO repos

- Much cleaner visualization focus (only final result — easier to understand convergence quality)
- Vectorized NumPy → faster execution
- User-friendly CLI input (great for teaching/demos)
- Better defaults tuned for multimodal functions like Rastrigin
- Reflection boundary prevents particles from escaping search space

## Requirements

- Python 3.8+
- NumPy
- Matplotlib

```bash
pip install numpy matplotlib
