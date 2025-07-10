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

# Quadratic approximation of cost
def cost_quadratic_approx(x, u, goal, gs_map, R=np.eye(2)*0.1, w_logbar=5.0):
    nx, nu = 4, 2
    Q = np.zeros((nx, nx))
    Rm = R
    q = np.zeros(nx)
    r = 2 * R @ u
    l = u @ R @ u

    pos = x[:2][None, :]
    logbar_cost, logbar_grad = gs_map.mahalanobis_logbarrier(pos, delta=0.5)

    # 위치에 대한 gradient 및 hessian 유사항 반영
    q[:2] += w_logbar * logbar_grad[0]
    Q[:2, :2] += w_logbar * np.eye(2)
    l += w_logbar * logbar_cost[0]

    return Q, Rm, q, r, l

def terminal_cost_approx(x, goal):
    nx = 4
    Qf = np.eye(nx)
    Qf[2:,2:] = 0  # only penalize position
    qf = 2 * Qf @ (x - np.hstack([goal, [0, 0]]))
    lf = (x[:2] - goal)**2 @ np.ones(2)
    return Qf, qf, lf

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
        Q, R, q, r, l = cost_quadratic_approx(x_seq[t], u_seq[t], goal, gs_map, w_logbar=0.01)
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
T = 100
dt = 0.1
x0 = np.array([0.0, 0.0, 0.0, 0.0])
goal = np.array([8.0, 8.0])

# Initial guess: straight-line trajectory
positions = np.linspace(x0[:2], goal, T+1)
velocities = np.gradient(positions, dt, axis=0)
x_seq = np.hstack([positions, velocities])
u_seq = np.zeros((T, 2))

# Gaussian map
mus = np.array([
    [4.0, 4.0],   # 중앙의 큰 장애물
    [2.0, 6.0],   # 왼쪽 위 방해 장애물
    [6.0, 2.0],   # 오른쪽 아래 방해 장애물
    [5.0, 8.0],   # 목표 근처에 얇은 벽 형태
    [3.0, 3.0],   # 경로 중간에 있는 회피 유도 장애물
    [4.0, 6.0],   # 중앙 세로 벽
])

covs = np.array([
    [[1.0, 0.0],     # 중앙 큰 장애물 (둥근)
     [0.0, 1.0]],
    
    [[0.3, 0.0],     # 좁은 타원형 장애물 (수직 장벽)
     [0.0, 1.0]],
    
    [[1.0, 0.0],     # 가로 긴 장애물
     [0.0, 0.3]],
    
    [[0.2, 0.0],     # 수직 얇은 벽
     [0.0, 1.2]],
    
    [[0.6, 0.2],     # 타원형으로 기울어진 장애물
     [0.2, 0.5]],
    
    [[0.2, 0.0],     # 좁은 수직 장벽
     [0.0, 2.0]],
])

alphas = np.array([
    1.0,  # 중앙
    0.8,  # 왼쪽 위
    0.7,  # 오른쪽 아래
    0.9,  # 목표 근처
    0.6,  # 회피 유도
    0.7,  # 수직 장벽
])
# Gaussian map 생성
gs_map = Gaussian2DMap(mus, covs, alphas)

# Iterative optimization
x_seq = x_seq.copy()  # 초기 trajectory (reference)
cost_prev = 1e10
best_cost = 1e10
best_x_seq = None
best_u_seq = None

for i in range(1000):
    x_seq, u_seq = ipddp_step(x0, x_seq, u_seq, goal, gs_map, T, dt, alpha=0.5)
    
    cost_now = np.sum([np.linalg.norm(u)**2 for u in u_seq]) + np.linalg.norm(x_seq[-1][:2] - goal)**2
    print(f"[Iter {i}] Cost: {cost_now:.4f}")
    
    if cost_now < best_cost:
        best_cost = cost_now
        best_x_seq = x_seq.copy()
        best_u_seq = u_seq.copy()

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

if best_x_seq is not None:
    plt.plot(best_x_seq[:, 0], best_x_seq[:, 1], c='magenta', lw=2, linestyle='--', label='Best Path')
    plt.scatter(best_x_seq[:, 0], best_x_seq[:, 1], c='magenta', s=10, alpha=0.5, label='Best Points', zorder=4)

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
