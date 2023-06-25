import numpy as np
from scipy.integrate import odeint

def lorenz_equations(state, t):
    x, y, z = state
    sigma = 10
    rho = 28
    beta = 8/3
    dx_dt = sigma * (y - x)
    dy_dt = x * (rho - z) - y
    dz_dt = x * y - beta * z
    return [dx_dt, dy_dt, dz_dt]

def lorenz_jacobian(state):
    x, y, z = state
    sigma = 10
    rho = 28
    beta = 8/3
    return [[-sigma, sigma, 0],
            [rho - z, -1, -x],
            [y, x, -beta]]

def linearized_equations(state, perturbation):
    return np.dot(lorenz_jacobian(state), perturbation)

def compute_lyapunov_exponent(trajectory, perturbed_trajectories):
    distances = np.zeros((len(perturbed_trajectories), len(trajectory)))
    for i in range(len(perturbed_trajectories)):
        for j in range(len(trajectory)):
            distances[i, j] = np.linalg.norm(trajectory[j] - perturbed_trajectories[i, j])
    distances /= np.linalg.norm(perturbation)
    lyapunov_exponent = np.mean(np.log(distances), axis=1) / (t[-1] - t[0])
    return lyapunov_exponent


def compute_perturbed_trajectories(state, perturbation, t):
    num_perturbations = len(perturbation)
    perturbed_trajectories = np.zeros((num_perturbations, len(t), len(state)))
    perturbed_trajectories[:, 0, :] = np.repeat([state], num_perturbations, axis=0) + perturbation

    for i in range(len(t) - 1):
        perturbation_matrices = perturbed_trajectories[:, i, :]
        perturbation_derivatives = linearized_equations(state, perturbation_matrices)
        perturbed_trajectories[:, i + 1, :] = perturbation_derivatives + perturbation_matrices

    return perturbed_trajectories



x0, y0, z0 = 0.1, 0.1, 0.1
initial_state = [x0, y0, z0]
T = 1000000
t = np.linspace(0, T, num=1000)  # 时间范围
perturbation = np.identity(3) * 1e-3  # 初始微扰

# 计算原始系统演化
trajectory = odeint(lorenz_equations, initial_state, t)

# 计算线性化方程演化
perturbed_trajectories = compute_perturbed_trajectories(initial_state, perturbation, t)

lyapunov_exponent = compute_lyapunov_exponent(trajectory, perturbed_trajectories)
print("Lyapunov exponent:", lyapunov_exponent)
