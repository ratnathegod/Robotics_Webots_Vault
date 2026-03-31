import numpy as np
import matplotlib.pyplot as plt
import math


class Node:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.parent = None


def get_nearest_node(tree, q_rand):
    min_dist = float('inf')
    nearest_node = None

    for node in tree:
        dist = math.hypot(node.x - q_rand.x, node.y - q_rand.y)
        if dist < min_dist:
            min_dist = dist
            nearest_node = node

    return nearest_node


def extend(q_near, q_rand, max_dist=0.5):
    dx = q_rand.x - q_near.x
    dy = q_rand.y - q_near.y
    dist = math.hypot(dx, dy)

    if dist == 0:
        q_new = Node(q_near.x, q_near.y)
        q_new.parent = q_near
        return q_new

    step = min(max_dist, dist)
    new_x = q_near.x + (dx / dist) * step
    new_y = q_near.y + (dy / dist) * step

    q_new = Node(new_x, new_y)
    q_new.parent = q_near
    return q_new


def is_collision_free(q_new, obstacles):
    for ox, oy, radius in obstacles:
        dist = math.hypot(q_new.x - ox, q_new.y - oy)
        if dist <= radius:
            return False
    return True


def plan_rrt():
    q_start = Node(0.0, 0.0)
    q_goal = Node(5, 5)

    obstacles = [
        (2.0, 2.0, 0.8),
        (1.5, 3.5, 0.6),
        (3.5, 1.5, 0.6),
        (4.0, 4.0, 0.5)
    ]

    max_dist = 0.4
    goal_bias = 0.01
    max_iter = 1000

    tree = [q_start]
    path = []

    plt.ion()
    _, ax = plt.subplots(figsize=(7, 7))

    for i in range(max_iter):
        if np.random.rand() < goal_bias:
            q_rand = Node(q_goal.x, q_goal.y)
        else:
            q_rand = Node(np.random.uniform(-1, 6), np.random.uniform(-1, 6))

        q_near = get_nearest_node(tree, q_rand)
        if q_near is None:
            continue

        q_new = extend(q_near, q_rand, max_dist)

        if is_collision_free(q_new, obstacles):
            tree.append(q_new)

            dist_to_goal = math.hypot(q_new.x - q_goal.x, q_new.y - q_goal.y)
            if dist_to_goal <= max_dist and is_collision_free(q_goal, obstacles):
                q_goal.parent = q_new
                tree.append(q_goal)

                curr = q_goal
                while curr is not None:
                    path.append([curr.x, curr.y])
                    curr = curr.parent
                path.reverse()

                print(f"Goal Reached in {i} iterations!")
                break

        if i % 10 == 0:
            draw_env(ax, tree, path, q_start, q_goal, obstacles, i)

    draw_env(ax, tree, path, q_start, q_goal, obstacles, i)
    if not path:
        print("Failed to find path.")
    plt.ioff()
    plt.show()


def draw_env(ax, tree, path, start, goal, obstacles, iteration=0):
    ax.clear()
    ax.set_xlim(-1, 6)
    ax.set_ylim(-1, 6)

    for (ox, oy, r) in obstacles:
        ax.add_patch(plt.Circle((ox, oy), r, color='gray', zorder=1))

    for node in tree:
        if node.parent is not None:
            ax.plot([node.x, node.parent.x], [
                    node.y, node.parent.y], 'k-', lw=0.5, zorder=2)

    if path:
        path_arr = np.array(path)
        ax.plot(path_arr[:, 0], path_arr[:, 1], 'r-',
                lw=3, label='RRT Path', zorder=3)

    ax.plot(start.x, start.y, 'go', markersize=10, label='Start', zorder=4)
    ax.plot(goal.x, goal.y, 'g*', markersize=15, label='Goal', zorder=4)

    ax.set_title(
        f"Rapidly-exploring Random Tree (RRT) | Iteration: {iteration}")
    ax.legend(loc='upper left')
    ax.grid(True)
    plt.pause(0.01)


if __name__ == '__main__':
    np.random.seed(42)
    plan_rrt()