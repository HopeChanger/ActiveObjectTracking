import os
import json
import numpy as np
import imageio
from copy import deepcopy
import matplotlib.pyplot as plt
from datetime import datetime
import matplotlib.patches as mpatches

from ENV.DigitalPose2D.target_move import GoalNavAgent


class MultiAgentEnv(object):
    def __init__(self, reset_type=0,
                 config_path="./settings/zq-PoseEnvBase.json",
                 setting_path=None,
                 render_save_path='./render'
                 ):
        """
        Load the environment settings.
        :param reset_type:
        :param config_path:
        :param setting_path:
        :param render_save_path:
        """

        self.reset_type = reset_type
        self.render_save_path = render_save_path

        self.ENV_PATH = os.path.dirname(os.path.abspath(__file__))
        if setting_path:
            self.SETTING_PATH = setting_path
        else:
            self.SETTING_PATH = os.path.join(self.ENV_PATH, config_path)
        with open(self.SETTING_PATH, encoding='utf-8') as f:
            setting = json.load(f)

        # camera/target init
        self.env_name = setting['env_name']
        self.cam_id = setting['cam_id']
        self.cam_num = len(self.cam_id)
        self.cam_area = np.array(setting['cam_area'])
        self.target_id = setting['targets']
        self.target_num = len(self.target_id)
        self.reset_area = setting['reset_area']
        # camera/target param
        self.visual_angle = setting['visual_angle']
        self.visual_distance = setting['visual_distance']
        self.target_moving_param = setting['target_moving_param']

        self.discrete_actions = setting['discrete_actions']
        self.rotation_scale = setting['rotation_scale']
        self.moving_scale = setting["moving_scale"]

        self.nav = setting['nav']

        self.max_steps = setting['max_steps']

        self.safe_start = setting['safe_start']
        self.start_area = self.safe_start[0]

        # random
        self.seed = None
        self.random_state = None

        self.cam = dict()
        for i in self.cam_id:
            self.cam[i] = dict(
                location=[0, 0],
                rotation=[0],
                trajectory=0
            )

        if 'Goal' in self.nav:
            self.random_agents = [GoalNavAgent(i, self.target_moving_param, self.reset_area)
                                  for i in range(self.target_num)]

        self.target_pos_list = None

        self.w = self.reset_area[1] - self.reset_area[0]
        self.h = self.reset_area[3] - self.reset_area[2]
        self.mean = ((self.reset_area[1] + self.reset_area[0]) / 2, (self.reset_area[3] + self.reset_area[2]) / 2)
        self.scale = max(self.w / 2, self.h / 2)
        self.max_traj = (self.w + self.h) * 2

    def set_seed(self, seed):
        self.seed = seed

    def random_reset(self):
        """Reset the environment."""
        np.random.seed(self.seed)
        # reset targets
        self.target_pos_list = np.array([[
            float(np.random.randint(self.start_area[0], self.start_area[1])),
            float(np.random.randint(self.start_area[2], self.start_area[3]))] for _ in range(self.target_num)])
        # reset agent
        for i in range(len(self.random_agents)):
            if 'Goal' in self.nav:
                self.random_agents[i].reset()

        # reset camera
        shuffle_cam_id = np.array(self.cam_id)
        np.random.shuffle(shuffle_cam_id)
        for i, cam in enumerate(shuffle_cam_id):
            if len(self.cam_area[i]) == 1:
                traj = self.cam_area[i][0]
            elif len(self.cam_area[i]) == 2:
                traj = np.random.randint(self.cam_area[i][0], self.cam_area[i][1])
            else:
                raise ValueError('len(cam_area[i]) should be 1 or 2.')
            self.reset_trajectory(cam)
            self.add_trajectory(cam, traj)
        for i, cam in enumerate(self.cam_id):
            cam_rot = np.random.randint(-179, 180) * 1.0
            self.set_rotation(cam, [cam_rot])
        self.random_state = np.random.get_state()
        np.random.seed(None)

    def fix_reset(self):
        np.random.seed(self.seed)
        # reset targets
        self.target_pos_list = np.array([[
            float(np.random.randint(self.start_area[0], self.start_area[1])),
            float(np.random.randint(self.start_area[2], self.start_area[3]))] for _ in range(self.target_num)])
        # reset agent
        for i in range(len(self.random_agents)):
            if 'Goal' in self.nav:
                self.random_agents[i].reset()

        # reset camera
        cam_rot = [-45, -90, -135, 135, 90, 45]
        for i, cam in enumerate(self.cam_id):
            if len(self.cam_area[i]) == 1:
                traj = self.cam_area[i][0]
            elif len(self.cam_area[i]) == 2:
                traj = np.random.randint(self.cam_area[i][0], self.cam_area[i][1])
            else:
                raise ValueError('len(cam_area[i]) should be 1 or 2.')
            self.reset_trajectory(cam)
            self.add_trajectory(cam, traj)
            self.set_rotation(cam, [cam_rot[i]])
        self.random_state = np.random.get_state()
        np.random.seed(None)

    def random_reset2(self):
        """Reset the environment."""
        np.random.seed(self.seed)
        # reset targets
        self.target_pos_list = np.array([[
            float(np.random.randint(self.start_area[0], self.start_area[1])),
            float(np.random.randint(self.start_area[2], self.start_area[3]))] for _ in range(self.target_num)])
        # reset agent
        for i in range(len(self.random_agents)):
            if 'Goal' in self.nav:
                self.random_agents[i].reset()

        # reset camera
        for i, cam in enumerate(self.cam_id):
            if len(self.cam_area[i]) == 1:
                traj = self.cam_area[i][0]
            elif len(self.cam_area[i]) == 2:
                traj = np.random.randint(self.cam_area[i][0], self.cam_area[i][1])
            else:
                raise ValueError('len(cam_area[i]) should be 1 or 2.')
            self.reset_trajectory(cam)
            self.add_trajectory(cam, traj)
        for i, cam in enumerate(self.cam_id):
            cam_rot = np.random.randint(-179, 180) * 1.0
            self.set_rotation(cam, [cam_rot])
        self.random_state = np.random.get_state()
        np.random.seed(None)

    def load_env_status(self, cam_info, target_info):
        # load targets information
        self.target_pos_list = deepcopy(target_info)

        # GoalNavAgent not load

        # load camera information
        self.cam = deepcopy(cam_info)

    def camera_move(self, actions):
        for i, cam in enumerate(self.cam_id):
            cam_rot = self.get_rotation(cam)
            cam_action = self.discrete_actions[actions[i]]
            cam_rot[0] += cam_action[0] * self.rotation_scale
            self.set_rotation(cam, cam_rot)
            self.add_trajectory(cam, cam_action[1] * self.moving_scale)

    def target_move(self, fix_step=None):
        delta_time = 0.13
        if 'Random' in self.nav:
            step_min = self.target_moving_param["random_step"][0]
            step_max = self.target_moving_param["random_step"][1]
            for i in range(self.target_num):
                self.target_pos_list[i] += [np.random.randint(step_min, step_max),
                                            np.random.randint(step_min, step_max)]
        elif 'Goal' in self.nav:
            for i in range(self.target_num):  # only one
                loc = list(self.target_pos_list[i])
                action = self.random_agents[i].act(loc)
                delta_x = action[1] * action[0] * delta_time
                delta_y = action[2] * action[0] * delta_time
                self.target_pos_list[i][0] += delta_x
                self.target_pos_list[i][1] += delta_y
            self.target_pos_list[:, 0] = np.clip(self.target_pos_list[:, 0], self.reset_area[0], self.reset_area[1])
            self.target_pos_list[:, 1] = np.clip(self.target_pos_list[:, 1], self.reset_area[2], self.reset_area[3])
        elif 'Fix' in self.nav:
            self.target_pos_list += fix_step
            self.target_pos_list[:, 0] = np.clip(self.target_pos_list[:, 0], self.reset_area[0], self.reset_area[1])
            self.target_pos_list[:, 1] = np.clip(self.target_pos_list[:, 1], self.reset_area[2], self.reset_area[3])

    def env_one_step(self, actions):
        np.random.set_state(self.random_state)
        self.target_move()
        self.camera_move(actions)
        self.random_state = np.random.get_state()
        np.random.seed(None)

    def render(self, file_name=None):
        """Render the environment."""
        status = self.get_target_status()

        camera_pos = []
        for i, cam in enumerate(self.cam_id):
            cam_pos = self.get_position(cam)
            camera_pos.append(cam_pos)

        camera_pos = np.array(camera_pos)
        target_pos = np.array(self.target_pos_list)

        num_cam = len(camera_pos)
        num_target = len(target_pos)

        pic_area = [
            self.reset_area[0] - self.visual_distance,
            self.reset_area[1] + self.visual_distance,
            self.reset_area[2] - self.visual_distance,
            self.reset_area[3] + self.visual_distance
        ]

        plt.figure(0, figsize=(5, 5))
        plt.cla()
        plt.axis("equal")
        plt.axis(pic_area)
        ax = plt.gca()

        plt.text(pic_area[0], pic_area[3], '{} sensors & {} targets'.format(num_cam, num_target), color="black")
        edge = plt.Rectangle(xy=(self.reset_area[0], self.reset_area[2]),
                             width=self.reset_area[1] - self.reset_area[0],
                             height=self.reset_area[3] - self.reset_area[2],
                             edgecolor="black", fill=False, linewidth=1)
        ax.add_artist(edge)
        for i in range(num_cam):
            a, b, theta = camera_pos[i]
            theta_range = np.arange(0, 2 * np.pi, 0.01)
            x = a + self.visual_distance * np.cos(theta_range)
            y = b + self.visual_distance * np.sin(theta_range)
            plt.plot(x, y, linestyle=' ', linewidth=1, color='steelblue', dashes=(6, 5.), dash_capstyle='round',
                     alpha=0.9)
            circle_c = plt.Circle((a, b), 4, color='slategray', fill=True)
            circle_bg = plt.Circle((a, b), self.visual_distance, color='steelblue', fill=True, alpha=0.05)
            wedge = mpatches.Wedge((a, b), self.visual_distance,
                                   theta - self.visual_angle / 2, theta + self.visual_angle / 2,
                                   color='green', alpha=0.2)
            plt.annotate(str(i + 1), xy=(a, b), xytext=(a, b), fontsize=10, color='black')
            ax.add_artist(circle_c)
            ax.add_artist(circle_bg)
            ax.add_artist(wedge)

        for i in range(num_target):
            if status[i] == 0:
                plt.plot(target_pos[i][0], target_pos[i][1], color='firebrick', marker="o", alpha=0.3)
                # plt.plot(self.noisy_target_pos_list[i][0], self.noisy_target_pos_list[i][1], color='dodgerblue', marker="o", alpha=0.3)
            else:
                plt.plot(target_pos[i][0], target_pos[i][1], color='firebrick', marker="o")
                # plt.plot(self.noisy_target_pos_list[i][0], self.noisy_target_pos_list[i][1], color='dodgerblue', marker="o")
            plt.annotate(str(i + 1), xy=(target_pos[i][0], target_pos[i][1]),
                         xytext=(target_pos[i][0], target_pos[i][1]), fontsize=10, color='maroon')

        plt.axis('off')
        # plt.show()
        if self.render_save_path:
            if file_name is None:
                file_name = '{}.jpg'.format(str(datetime.now()).replace('.', '-').replace(':', '-'))
            if not os.path.exists(self.render_save_path):
                os.makedirs(self.render_save_path)
            plt.savefig(os.path.join(self.render_save_path, file_name))
        plt.pause(0.01)

    def create_gif(self, gif_name='output.gif'):
        if self.render_save_path is None:
            raise ValueError('No saved pictures.')
        image_list = [os.path.join(self.render_save_path, img) for img in os.listdir(self.render_save_path)]
        image_list.sort()
        frames = []
        for image_name in image_list:
            if image_name.endswith('.jpg'):
                frames.append(imageio.imread(image_name))
        # Save them as frames into a gif
        imageio.mimsave(os.path.join(self.render_save_path, gif_name), frames, 'GIF', duration=0.2)

    def set_location(self, cam_id, loc):
        # assert type(loc, list)
        self.cam[cam_id]['location'] = loc.copy()

    def get_location(self, cam_id):
        return self.cam[cam_id]['location'].copy()

    def set_rotation(self, cam_id, rot):
        # -180~180(include) X-axis is 0
        rot = rot.copy()
        for i in range(len(rot)):
            if rot[i] > 180:
                rot[i] -= 360
            if rot[i] <= -180:
                rot[i] += 360
        self.cam[cam_id]['rotation'] = rot

    def get_rotation(self, cam_id):
        return self.cam[cam_id]['rotation'].copy()

    def set_position(self, cam_id, pos):
        self.set_location(cam_id, pos[:2])
        self.set_rotation(cam_id, pos[2:])

    def get_position(self, cam_id):
        loc = self.get_location(cam_id)
        rot = self.get_rotation(cam_id)
        return loc + rot

    def add_trajectory(self, cam_id, traj):
        new_traj = self.cam[cam_id]['trajectory'] + traj
        if new_traj >= self.max_traj:
            new_traj -= self.max_traj
        elif new_traj < 0:
            new_traj += self.max_traj
        self.cam[cam_id]['trajectory'] = new_traj
        if new_traj < self.w:
            self.set_location(cam_id, [self.reset_area[0] + new_traj, self.reset_area[3]])
        elif new_traj < self.w + self.h:
            self.set_location(cam_id, [self.reset_area[1], self.reset_area[3] - (new_traj - self.w)])
        elif new_traj < 2 * self.w + self.h:
            self.set_location(cam_id, [self.reset_area[1] - (new_traj - self.w - self.h), self.reset_area[2]])
        else:
            self.set_location(cam_id, [self.reset_area[0], self.reset_area[2] + (new_traj - 2 * self.w - self.h)])

    def reset_trajectory(self, cam_id):
        self.cam[cam_id]['trajectory'] = 0

    def visible(self, angle, dist):
        if dist <= self.visual_distance and abs(angle) <= self.visual_angle / 2:
            return True
        else:
            return False

    def get_target_status(self):
        """
        0: invisible, 1: visible
        """
        see_times = np.zeros(self.target_num)
        for i, cam in enumerate(self.cam_id):
            cam_pos = self.get_position(cam)
            for j in range(self.target_num):
                angle, dist = self.get_relative_position(cam_pos, self.target_pos_list[j])
                if self.visible(angle, dist):
                    see_times[j] += 1
        status = np.zeros(self.target_num)
        status[np.where(see_times > 0)[0]] = 1
        return status

    @staticmethod
    def get_relative_position(current_pose, target_pose):
        y_delt = target_pose[1] - current_pose[1]
        x_delt = target_pose[0] - current_pose[0]
        angle = np.arctan2(y_delt, x_delt) / np.pi * 180 - current_pose[2]
        dist = np.sqrt(y_delt * y_delt + x_delt * x_delt)
        if angle > 180:
            angle -= 360
        if angle < -180:
            angle += 360
        return angle, dist
