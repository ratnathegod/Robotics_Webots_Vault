import numpy as np
import matplotlib.pyplot as plt


def compute_gradient(q, q_goal, q_obs, k_att=10.0, k_rep=5.0):
    grad_att = k_att * (q - q_goal)

    dist = np.linalg.norm(q - q_obs)
    grad_rep = -k_rep * (q - q_obs) / (dist ** 4)

    return grad_att + grad_rep


def animate_potential_field():
    q_start = np.array([0.0, 0.0])
    q_goal = np.array([4.0, 5.0])
    q_obs = np.array([2.0, 2.0])

    alpha = 0.01
    max_steps = 100

    q_current = np.copy(q_start)
    trajectory = [np.copy(q_current)]

    # Setup Animation
    plt.ion()
    _, ax = plt.subplots(figsize=(7, 7))

    for step in range(max_steps):
        grad = compute_gradient(q_current, q_goal, q_obs)

        q_current = q_current - alpha * grad

        trajectory.append(np.copy(q_current))

        ax.clear()
        ax.set_xlim(-1, 6)
        ax.set_ylim(-1, 6)

        ax.plot(q_start[0], q_start[1], 'go', markersize=10, label='Start')
        ax.plot(q_goal[0], q_goal[1], 'g*', markersize=15, label='Goal')
        ax.plot(q_obs[0], q_obs[1], 'ro', markersize=12, label='Obstacle')

        traj_arr = np.array(trajectory)
        ax.plot(traj_arr[:, 0], traj_arr[:, 1], 'b--', label='Trajectory')
        ax.plot(q_current[0], q_current[1], 'bo', markersize=8, label='Robot')

        ax.set_title(f"Gradient Descent Navigation | Step: {step}")
        ax.legend(loc='upper left')
        ax.grid(True)

        plt.pause(0.05)


        if np.linalg.norm(q_current - q_goal) < 0.1:
            print(f"Goal Reached in {step} steps!")
            break

    plt.ioff()
    plt.show()


if __name__ == "__main__":
    animate_potential_field()