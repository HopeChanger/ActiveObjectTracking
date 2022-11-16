import numpy as np


class GoalNavAgent(object):
    def __init__(self, id, action_space, goal_area, goal_list=None):
        self.id = id
        self.velocity_low = action_space['velocity'][0]
        self.velocity_high = action_space['velocity'][1]
        self.angle_low = action_space['angle'][0]
        self.angle_high = action_space['angle'][1]
        self.goal_area = goal_area

        self.step_counter = 0
        self.goal_id = 0
        self.max_len = 100
        self.goal_list = goal_list

        self.goal = self.generate_goal()
        self.velocity = np.random.randint(self.velocity_low, self.velocity_high)
        self.pose_last = [[], []]

    def act(self, pose):
        self.step_counter += 1
        if len(self.pose_last[0]) == 0:
            self.pose_last[0] = np.array(pose)
            self.pose_last[1] = np.array(pose)
            d_moved = 30
        else:
            d_moved = min(np.linalg.norm(np.array(self.pose_last[0]) - np.array(pose)),
                          np.linalg.norm(np.array(self.pose_last[1]) - np.array(pose)))
            self.pose_last[0] = np.array(self.pose_last[1])
            self.pose_last[1] = np.array(pose)
        if self.check_reach(self.goal, pose) or d_moved < 10 or self.step_counter > self.max_len:
            self.goal = self.generate_goal()
            self.velocity = np.random.randint(self.velocity_low, self.velocity_high)
            self.step_counter = 0

        delt_unit = (self.goal[:2] - pose[:2]) / np.linalg.norm(self.goal[:2] - pose[:2])
        velocity = self.velocity * (1 + 0.2 * np.random.random())
        return [velocity, delt_unit[0], delt_unit[1]]

    def reset(self):
        self.step_counter = 0
        self.goal_id = 0
        self.goal = self.generate_goal()
        self.velocity = np.random.randint(self.velocity_low, self.velocity_high)
        self.pose_last = [[], []]

    def generate_goal(self):
        if self.goal_list and len(self.goal_list) != 0:
            index = self.goal_id % len(self.goal_list)
            goal = np.array(self.goal_list[index])
        else:
            x = np.random.randint(self.goal_area[0], self.goal_area[1])
            y = np.random.randint(self.goal_area[2], self.goal_area[3])
            goal = np.array([x, y])
        self.goal_id += 1
        return goal

    def check_reach(self, goal, now):
        error = np.array(now[:2]) - np.array(goal[:2])
        distance = np.linalg.norm(error)
        return distance < 5
