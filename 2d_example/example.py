import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv
from gaussian_2d import Gaussian2DMap

# Dynamics: [x, y, vx, vy], u = [ax, ay]
def dynamics(x, u, dt):
    return np.array([
        x[0] + x[2]*dt,
        x[1] + x[3]*dt,
        x[2] + u[0]*dt,
        x[3] + u[1]*dt
    ])

# Finite Difference for fx, fu
def finite_difference_jacobian(f, x, u, dt, eps=1e-4):
    nx, nu = x.shape[0], u.shape[0]
    fx = np.zeros((nx, nx))
    fu = np.zeros((nx, nu))
    for i in range(nx):
        dx = np.zeros(nx)
        dx[i] = eps
        fx[:, i] = (f(x + dx, u, dt) - f(x - dx, u, dt)) / (2 * eps)
    for i in range(nu):
        du = np.zeros(nu)
        du[i] = eps
        fu[:, i] = (f(x, u + du, dt) - f(x, u - du, dt)) / (2 * eps)
    return fx, fu

# Build cost function from costmap (bilinear interpolation)
def build_cost_fn_from_costmap(costmap, grid_x, grid_y):
    def fn(x):
        px, py = x[0], x[1]
        ix = np.clip(np.searchsorted(grid_x, px) - 1, 0, len(grid_x) - 2)
        iy = np.clip(np.searchsorted(grid_y, py) - 1, 0, len(grid_y) - 2)

        x1, x2 = grid_x[ix], grid_x[ix+1]
        y1, y2 = grid_y[iy], grid_y[iy+1]
        q11 = costmap[iy, ix]
        q12 = costmap[iy+1, ix]
        q21 = costmap[iy, ix+1]
        q22 = costmap[iy+1, ix+1]

        denom = (x2 - x1) * (y2 - y1)
        if denom == 0:
            return 0
        interp = (q11 * (x2 - px) * (y2 - py) +
                  q21 * (px - x1) * (y2 - py) +
                  q12 * (x2 - px) * (py - y1) +
                  q22 * (px - x1) * (py - y1)) / denom
        return interp
    return fn

# Quadratic approximation of cost
def cost_quadratic_approx(x, u, goal, gs_map, R=np.eye(2)*0.1, w_obs=1.0, w_risk=2.0):
    nx, nu = 4, 2
    Q = np.zeros((nx, nx))
    Rm = R
    q = np.zeros(nx)
    r = 2 * R @ u
    l = u @ R @ u

    pos = x[:2][None, :] 
    _, _, weighted_risk = gs_map.density_with_cov(pos)
    l += w_risk * weighted_risk[0] 

    return Q, Rm, q, r, l

def terminal_cost_approx(x, goal):
    nx = 4
    Qf = np.eye(nx)
    Qf[2:,2:] = 0  # only penalize position
    qf = 2 * Qf @ (x - np.hstack([goal, [0, 0]]))
    lf = (x[:2] - goal)**2 @ np.ones(2)
    return Qf, qf, lf

# Build local costmap from trajectory
def build_local_costmap_from_trajectory(x_seq, density_map, grid_x, grid_y):
    # Clip density for safety
    return density_map

# IPDDP core step
def ipddp_step(x0, x_ref, u_seq, goal, gs_map, T, dt, alpha=0.5):
    nx, nu = 4, 2

    # Forward rollout to get current state sequence
    x_seq = [x0]
    for u in u_seq:
        x_seq.append(dynamics(x_seq[-1], u, dt))
    x_seq = np.array(x_seq)

    # DDP backward pass
    fx_seq, fu_seq = [], []
    Q_seq, R_seq, q_seq, r_seq, l_seq = [], [], [], [], []

    for t in range(T):
        fx, fu = finite_difference_jacobian(dynamics, x_seq[t], u_seq[t], dt)
        Q, R, q, r, l = cost_quadratic_approx(x_seq[t], u_seq[t], goal, gs_map, w_obs=50.0)
        fx_seq.append(fx)
        fu_seq.append(fu)
        Q_seq.append(Q)
        R_seq.append(R)
        q_seq.append(q)
        r_seq.append(r)
        l_seq.append(l)

    Qf, qf, lf = terminal_cost_approx(x_seq[-1], goal)

    V = Qf
    v = qf
    k_seq = []
    K_seq = []

    for t in reversed(range(T)):
        fx, fu = fx_seq[t], fu_seq[t]
        Q, R = Q_seq[t], R_seq[t]
        q, r = q_seq[t], r_seq[t]

        Q_x = Q + fx.T @ V @ fx
        Q_u = R + fu.T @ V @ fu
        Q_xu = fx.T @ V @ fu
        Q_ux = Q_xu.T
        q_x = q + fx.T @ v
        q_u = r + fu.T @ v

        inv_Q_u = inv(Q_u + 1e-5 * np.eye(nu))
        K = -inv_Q_u @ Q_ux
        k = -inv_Q_u @ q_u

        V = Q_x + K.T @ Q_u @ K + K.T @ Q_ux + Q_ux.T @ K
        v = q_x + K.T @ Q_u @ k + K.T @ q_u + Q_ux.T @ k

        k_seq.insert(0, k)
        K_seq.insert(0, K)

    # Forward rollout with alpha step size
    x = x0
    x_seq_new = [x0]
    u_seq_new = []
    for t in range(T):
        dx = x - x_seq[t]
        u = u_seq[t] + alpha * k_seq[t] + K_seq[t] @ dx
        x = dynamics(x, u, dt)
        x_seq_new.append(x)
        u_seq_new.append(u)

    return np.array(x_seq_new), np.array(u_seq_new)

# -------------------------
# Main Execution
# -------------------------
T = 50
dt = 0.2
x0 = np.array([0.0, 0.0, 0.0, 0.0])
goal = np.array([8.0, 8.0])

# Initial guess: straight-line trajectory
positions = np.linspace(x0[:2], goal, T+1)
velocities = np.gradient(positions, dt, axis=0)
x_seq = np.hstack([positions, velocities])
u_seq = np.zeros((T, 2))

# Gaussian map
mus = np.array([
    [2.0, 3.0],
    [5.0, 5.0],
    [7.0, 2.0],
    [7.5, 7.5]
])

covs = np.array([
    [[0.5, 0.2],
     [0.2, 0.3]],
    [[0.3, 0.0],
     [0.0, 0.3]],
    [[0.7, -0.1],
     [-0.1, 0.4]],
    [[0.7, -0.5],
     [-0.5, 0.7]],
])

alphas = np.array([1.0, 0.8, 0.6, 0.9])

# Gaussian map 생성
gs_map = Gaussian2DMap(mus, covs, alphas)

# Iterative optimization
x_seq = x_seq.copy()  # 초기 trajectory (reference)
cost_prev = 1e10
for i in range(1000):
    x_seq, u_seq = ipddp_step(x0, x_seq, u_seq, goal, gs_map, T, dt, alpha=0.5)
    
    cost_now = np.sum([np.linalg.norm(u)**2 for u in u_seq]) + np.linalg.norm(x_seq[-1][:2] - goal)**2
    print(f"[Iter {i}] Cost: {cost_now:.4f}")
    
    if abs(cost_now - cost_prev) < 1e-3 and np.linalg.norm(u_seq - u_seq_prev) < 1e-3:
        print("Converged.")
        break
    cost_prev = cost_now
    u_seq_prev = u_seq

x = np.linspace(0, 10, 200)
y = np.linspace(0, 10, 200)
xx, yy = np.meshgrid(x, y)
pts = np.stack([xx.ravel(), yy.ravel()], axis=1)
density_map = gs_map.density(pts).reshape(xx.shape)

# Plot result
plt.figure(figsize=(8, 8))
plt.contourf(xx, yy, density_map, levels=50, cmap='inferno')  # 배경 density

# Trajectory
plt.plot(x_seq[:, 0], x_seq[:, 1], c='cyan', lw=3, label='Optimized Path')
plt.scatter(x_seq[:, 0], x_seq[:, 1], c='white', s=15, label='Trajectory Points', zorder=5)

# Start and goal
plt.scatter(x0[0], x0[1], c='blue', label='Start', s=50, edgecolors='k')
plt.scatter(goal[0], goal[1], c='green', label='Goal', s=50, edgecolors='k')

plt.legend()
plt.title("IPDDP Optimized Trajectory over GS Density")
plt.xlabel("X")
plt.ylabel("Y")
plt.axis('equal')
plt.grid(True)
plt.tight_layout()
plt.show()
