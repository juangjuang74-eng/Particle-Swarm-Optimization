# pso_swarm_visual_only.py
# Focus: Interactive PSO → show ONLY final swarm positions on function landscape
# Improved visual: clear contour background + particles + global best

import numpy as np
import matplotlib.pyplot as plt


def sphere(x):
    return np.sum(x**2, axis=1)


def rastrigin(x):
    A = 10
    return A * x.shape[1] + np.sum(x**2 - A * np.cos(2 * np.pi * x), axis=1)


def rosenbrock(x):
    return np.sum(100 * (x[:, 1:] - x[:, :-1]**2)**2 + (1 - x[:, :-1])**2, axis=1)


FUNCTIONS = {
    'sphere':     (sphere,     -5.12,  5.12),
    'rastrigin':  (rastrigin,  -5.12,  5.12),
    'rosenbrock': (rosenbrock, -2.048, 2.048),
}


def get_user_input():
    print("\n=== PSO Swarm Visualizer (final positions only) ===\n")
    
    func_name = input("Function (sphere / rastrigin / rosenbrock) [default: rastrigin]: ").strip().lower()
    if not func_name or func_name not in FUNCTIONS:
        func_name = 'rastrigin'
    
    while True:
        try:
            dim = int(input("Dimension [default: 2 for visualization]: ") or 2)
            if dim < 1:
                raise ValueError
            break
        except:
            print("Please enter a positive integer.")
    
    if dim != 2:
        print("Note: Detailed swarm plot is best viewed in 2D. Higher dimensions will still run but visualization is skipped.")
    
    n_particles = int(input("Number of particles [default: 60]: ") or 60)
    max_iter    = int(input("Max iterations [default: 120]: ") or 120)
    
    print("\nUsing:")
    print(f"  Function     : {func_name}")
    print(f"  Dimension    : {dim}")
    print(f"  Particles    : {n_particles}")
    print(f"  Max iter     : {max_iter}")
    print("  w_max = 0.9 → w_min = 0.39 (linear decay)")
    print("  c1 = c2 = 2.05\n")
    
    return func_name, dim, n_particles, max_iter


class SimplePSO:
    def __init__(self, func, lb, ub, dim, n_particles, max_iter):
        self.func = func
        self.lb = lb
        self.ub = ub
        self.dim = dim
        self.n_particles = n_particles
        self.max_iter = max_iter
        
        self.w_max = 0.9
        self.w_min = 0.39
        self.c1 = 2.05
        self.c2 = 2.05
        self.v_max_factor = 0.18
        
        self.search_range = ub - lb
        self.v_max = self.v_max_factor * self.search_range
        
        # Initialize
        self.x = np.random.uniform(lb, ub, (n_particles, dim))
        self.v = np.random.uniform(-self.v_max, self.v_max, (n_particles, dim))
        
        self.p_best = self.x.copy()
        self.p_best_score = self.func(self.p_best)
        
        self.g_best_idx = np.argmin(self.p_best_score)
        self.g_best = self.p_best[self.g_best_idx].copy()
        self.g_best_score = self.p_best_score[self.g_best_idx]
    
    def run(self, verbose=True):
        for it in range(self.max_iter):
            w = self.w_max - (self.w_max - self.w_min) * (it / self.max_iter)
            
            r1 = np.random.rand(self.n_particles, self.dim)
            r2 = np.random.rand(self.n_particles, self.dim)
            
            cognitive = self.c1 * r1 * (self.p_best - self.x)
            social    = self.c2 * r2 * (self.g_best - self.x)
            self.v = w * self.v + cognitive + social
            self.v = np.clip(self.v, -self.v_max, self.v_max)
            
            self.x += self.v
            
            # Simple reflection boundary
            outside_low  = self.x < self.lb
            outside_high = self.x > self.ub
            self.x[outside_low]  = 2 * self.lb  - self.x[outside_low]
            self.x[outside_high] = 2 * self.ub  - self.x[outside_high]
            self.v[outside_low | outside_high] *= -0.4
            
            scores = self.func(self.x)
            
            improved = scores < self.p_best_score
            self.p_best_score[improved] = scores[improved]
            self.p_best[improved] = self.x[improved].copy()
            
            if np.min(scores) < self.g_best_score:
                self.g_best_idx = np.argmin(scores)
                self.g_best_score = scores[self.g_best_idx]
                self.g_best = self.x[self.g_best_idx].copy()
            
            if verbose and it % 30 == 0:
                print(f"Iter {it:4d} | w={w:.3f} | best={self.g_best_score:.6e}")
        
        return self.x, self.g_best, self.g_best_score


def plot_final_swarm(positions, best_pos, func, lb, ub, func_name):
    if positions.shape[1] != 2:
        print("\nVisualization skipped: swarm plot only works in 2 dimensions.")
        return
    
    # Generate contour background
    margin = 0.1 * (ub - lb)
    x = np.linspace(lb - margin, ub + margin, 180)
    y = np.linspace(lb - margin, ub + margin, 180)
    X, Y = np.meshgrid(x, y)
    Z = func(np.column_stack([X.ravel(), Y.ravel()])).reshape(X.shape)
    
    plt.figure(figsize=(10, 8))
    
    # Contour with logarithmic levels if needed
    if Z.min() > 0:
        levels = np.logspace(np.log10(Z.min()), np.log10(Z.max()), 30)
    else:
        levels = 30
    
    plt.contourf(X, Y, Z, levels=levels, cmap='magma', alpha=0.85)
    plt.colorbar(label='Function value')
    
    # Final particle positions
    plt.scatter(positions[:,0], positions[:,1],
                c='white', edgecolors='black', s=70, linewidth=0.8,
                label='Particles (final)', zorder=5, alpha=0.9)
    
    # Global best
    plt.scatter(best_pos[0], best_pos[1],
                c='lime', marker='*', s=380, edgecolors='black', linewidth=1.5,
                label='Global Best', zorder=10)
    
    plt.title(f"Final PSO Swarm – {func_name} (dim=2)\nBest value = {best_score:.6e}", fontsize=13)
    plt.xlabel("x₁")
    plt.ylabel("x₂")
    plt.legend(loc='upper right', fontsize=10)
    plt.grid(alpha=0.25, linestyle='--')
    plt.axis('equal')
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    func_name, dim, n_particles, max_iter = get_user_input()
    
    func, lb, ub = FUNCTIONS[func_name]
    
    print("\nRunning PSO...\n")
    
    optimizer = SimplePSO(func, lb, ub, dim, n_particles, max_iter)
    final_positions, best_position, best_score = optimizer.run(verbose=True)
    
    print("\n" + "="*50)
    print(f"Optimization finished!")
    print(f"Best position: {best_position}")
    print(f"Best value    : {best_score:.8e}")
    print("="*50 + "\n")
    
    plot_final_swarm(final_positions, best_position, func, lb, ub, func_name.upper())