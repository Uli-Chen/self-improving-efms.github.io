import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import matplotlib.patches as patches
import dm_env
from dm_env import specs
from typing import Optional, Dict, Any

BOUNDS_X = np.array([-1., 1.], dtype=np.float32)
BOUNDS_Y = np.array([-1., 1.], dtype=np.float32)

DPI = 200
RENDER_HEIGHT_INCHES = 5

class Point2D(dm_env.Environment):
    def __init__(self, physics_substeps=10, success_radius=0.15):
        self._cur_pos = np.zeros(2, dtype=np.float32)
        self._goal_pos = np.zeros(2, dtype=np.float32)
        self._cur_vel = np.zeros(2, dtype=np.float32)
        self._cur_episode_traj = []
        self._physics_substeps = physics_substeps
        self._success_radius = success_radius

    def sample_goal(self):
        border_x = (BOUNDS_X[1] - BOUNDS_X[0]) * 0.05
        border_y = (BOUNDS_Y[1] - BOUNDS_Y[0]) * 0.05
        goal_x = np.random.uniform(
            BOUNDS_X[0] + border_x, BOUNDS_X[1] - border_x)
        goal_y = np.random.uniform(
            BOUNDS_Y[0] + border_y, BOUNDS_Y[1] - border_y)
        return np.array([goal_x, goal_y], dtype=np.float32)

    def set_goal(self, goal_pos):
        self._goal_pos = goal_pos

    def reset(self):
        self._goal_pos = self.sample_goal()
        cur_x = np.random.uniform(BOUNDS_X[0], BOUNDS_X[1])
        cur_y = np.random.uniform(BOUNDS_Y[0], BOUNDS_Y[1])
        self._cur_pos = np.array([cur_x, cur_y], dtype=np.float32)
        cur_pos_copy = self._cur_pos.copy()

        self._cur_vel = np.zeros(2, dtype=np.float32)
        cur_vel_copy = self._cur_vel.copy()

        obs = {
            'cur_pos': cur_pos_copy,
            'cur_vel': cur_vel_copy,
            'goal_pos': self._goal_pos.copy()}
        ts = dm_env.TimeStep(
            step_type=dm_env.StepType.FIRST,
            reward=0.0, # changed from None for consistency with RL loops
            discount=1.0,
            observation=obs)

        self._cur_episode_traj = [cur_pos_copy]
        return ts

    def step(self, action):
        for _ in range(self._physics_substeps):
            self._cur_vel += action
            self._cur_pos += self._cur_vel

        cur_pos_copy = self._cur_pos.copy()
        cur_vel_copy = self._cur_vel.copy()
        obs = {
            'cur_pos': cur_pos_copy,
            'cur_vel': cur_vel_copy,
            'goal_pos': self._goal_pos.copy()}

        if self.success():
            step_type = dm_env.StepType.LAST
        else:
            step_type = dm_env.StepType.MID
            
        ts = dm_env.TimeStep(
            step_type=step_type,
            reward=-1. * np.linalg.norm(self._cur_pos - self._goal_pos),
            discount=1.,
            observation=obs,)

        self._cur_episode_traj.append(cur_pos_copy)
        return ts

    def success(self, waypoint: Optional[np.ndarray] = None):
        if waypoint is not None:
            goal_pos = waypoint
        else:
            goal_pos = self._goal_pos
        return np.linalg.norm(self._cur_pos - goal_pos) < self._success_radius

    def observation_spec(self):
        return {
            'cur_pos': specs.Array((2,), dtype=np.float32),
            'cur_vel': specs.Array((2,), dtype=np.float32),
            'goal_pos': specs.Array((2,), dtype=np.float32),}

    def action_spec(self):
        return specs.Array((2,), dtype=np.float32)

    def render(
        self,
        title: str = '',
        points: Optional[np.ndarray] = None,
        goal_pos: Optional[np.ndarray] = None):
        fig, ax = plt.subplots(
            figsize=(RENDER_HEIGHT_INCHES, RENDER_HEIGHT_INCHES), dpi=DPI)
        ax.set_xlim(BOUNDS_X[0], BOUNDS_X[1])
        ax.set_ylim(BOUNDS_Y[0], BOUNDS_Y[1])
        ax.set_aspect('equal')

        if points is None:
            points = np.array(self._cur_episode_traj)
            cur_pos = self._cur_pos
        else:
            cur_pos = points[-1]

        if goal_pos is None:
            goal_pos = self._goal_pos

        ax.plot(points[:, 0], points[:, 1], marker='.', color='blue', markersize=16, linewidth=4)
        ax.scatter(
            goal_pos[0], goal_pos[1], marker='*', s=200, color='orange', linewidths=8)
        ax.scatter(
            cur_pos[0], cur_pos[1], marker='o', s=100, color='red', linewidths=8)

        # Add a dashed circle around the star
        circle = patches.Circle(
            (goal_pos[0], goal_pos[1]),  # Center of the circle
            self._success_radius,  # Radius of the circle
            edgecolor='green',  # Color of the circle
            linestyle='--',  # Dashed line
            linewidth=4,  # Thickness of the circle line
            fill=False  # Ensure it's just an outline
        )
        ax.add_patch(circle)  # Add the circle to the plot

        # Make the axes lines thicker
        for spine in ax.spines.values():
            spine.set_linewidth(4)  # Adjust the thickness here

        if title != '':
            ax.set_title(title, fontsize=18, fontweight='bold')

        ax.set_xticks([])
        ax.set_yticks([])
        plt.tight_layout()

        # Render the plot using FigureCanvasAgg
        canvas = FigureCanvas(fig)
        canvas.draw()

        # Convert the rendered image to a numpy array
        width, height = fig.get_size_inches() * fig.get_dpi()

        buf = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8)
        image = buf.reshape(int(height), int(width), 4)  # RGBA
        image = image[:, :, :3]  # 去掉 alpha，转成 RGB

        plt.close(fig)
        return image
