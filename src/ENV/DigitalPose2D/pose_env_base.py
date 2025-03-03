import numpy as np
from copy import deepcopy
from functools import partial

from ENV.DigitalPose2D.multi_agent_env import MultiAgentEnv


class PoseEnvBase(MultiAgentEnv):
    def __init__(self, reset_type=0,
                 config_path="./settings/zq-PoseEnvBase.json",
                 setting_path=None,
                 render_save_path='./render',
                 args=None
                 ):
        super(PoseEnvBase, self).__init__(reset_type, config_path, setting_path, render_save_path)
        self.args = args
        self.predict_model = partial(PredictEnv, reset_type, config_path, setting_path, render_save_path, args)

        self.count_steps = 0

        self.memory = []
        self.memory_max_len = 10

    def get_env_info(self):
        env_info = {"n_actions": self.get_num_actions(),
                    "n_agents": self.get_num_agents(),
                    "episode_limit": self.get_episode_len(),
                    "centralized_obs_shape": self.get_centralized_obs_size()}
        return env_info

    def step(self, actions):
        """A single environment step. 
        actions: a list of size [n_agents], actions[i] is the action taken by the i^th agent
        Returns reward, team_reward, done.
        reward大小为[n_agents]， reward[i]为第i个agent的个人奖励"""
        status = self.get_target_status()
        self.memory.append([deepcopy(self.cam), deepcopy(self.target_pos_list), deepcopy(status)])
        if len(self.memory) > self.memory_max_len:
            del self.memory[0]

        actions = [int(a) for a in actions]
        
        # target/camera move
        self.env_one_step(actions)

        # reward
        reward, team_reward = self.get_reward()

        # render
        if self.args.render:
            self.render(file_name="{:03d}.jpg".format(self.count_steps))

        done = False
        self.count_steps += 1
        if self.count_steps >= self.max_steps:
            done = True

        return reward, team_reward, done

    def get_reward(self):
        reward = np.zeros(self.cam_num)
        cam_target = np.zeros((self.cam_num, self.target_num))
        for i, cam in enumerate(self.cam_id):
            cam_pos = self.get_position(cam)
            for j in range(self.target_num):
                angle, dist = self.get_relative_position(cam_pos, self.target_pos_list[j])
                if self.visible(angle, dist):
                    cam_target[i][j] = 1
        repeat = np.int64(cam_target.sum(axis=0) > 1)
        del_repeat = np.int64(cam_target - repeat > 0)
        for i, cam in enumerate(self.cam_id):
            reward[i] = sum(del_repeat[i]) / self.target_num
        team_reward = np.sum(np.sum(cam_target, 0) > 0) / self.target_num
        return reward, team_reward
    
    def reset(self):
        """Reset the environment."""
        self.count_steps = 0

        self.random_reset()

        self.memory = []

    def get_centralized_obs(self):
        """Returns centralized observations of all agent in a list. size: [n_agents, centralized_obs_shape]
        [d, sin(theta), cos(theta)]*m  x,y, sin(\gamma) cos(\gamma)
        首先将每个agent的去中心化obs后面添加一个长度为n_agents的onehot向量, 用于表示该agent的index, 如第一个agent就添加[1, 0,..., 0, 0]
        然后每个agent的中心化obs就是将该agent的去中心化obs放在最前面，后面添加上其他所有agent的去中心obs
        """
        cobs = []
        for i, cam in enumerate(self.cam_id):
            obs = []
            cam_pos = self.get_position(cam)
            for j in range(self.target_num):
                angle, dist = self.get_relative_position(cam_pos, self.target_pos_list[j])
                if self.visible(angle, dist):
                    obs += [dist / self.scale, np.sin(angle / 180 * np.pi), np.cos(angle / 180 * np.pi)]
                else:
                    obs += [-1, -1, -1]
            obs += [(cam_pos[0] - self.mean[0]) / self.scale, (cam_pos[1] - self.mean[1]) / self.scale,
                    np.sin(cam_pos[2] / 180 * np.pi), np.cos(cam_pos[2] / 180 * np.pi)]
            one_hot = [0] * self.cam_num
            one_hot[i] = 1
            obs += one_hot
            cobs.append(np.array(obs))

        centralized_obs = []
        for i in range(self.cam_num):
            now_obs = cobs[i].copy()
            for j in range(self.cam_num):
                if j != i:
                    now_obs = np.append(now_obs, cobs[j])
            # now_obs = np.append(now_obs, self.count_steps / self.max_steps)
            centralized_obs.append(now_obs)
        return centralized_obs

    def get_centralized_obs_size(self):
        """Returns the size of the centralized observation."""
        obs_size = (self.target_num * 3 + 4 + self.cam_num) * self.cam_num
        # assert len(self.get_centralized_obs()[0]) == obs_size
        return obs_size

    def get_avail_actions(self):
        """Returns the available actions of all agents in a list.
        If all actions are legal, return a full 1 matrix of size [n_agents, n_actions]"""
        return np.ones((self.get_num_agents(), self.get_num_actions()))

    def get_num_actions(self):
        """Returns the number of actions an agent could take."""
        return len(self.discrete_actions)
    
    def get_num_agents(self):
        """Returns the number of agents."""
        return self.cam_num
    
    def get_episode_len(self):
        """Returns the length of one episode."""
        return self.max_steps

    def get_predict_model(self):
        default_info = dict()
        now_status = self.get_target_status()
        if len(self.memory) == 0:
            default_info['default_pos'] = self.target_pos_list.copy()
            for i in range(self.target_num):
                default_info['default_pos'][i] = np.array([2 * self.reset_area[1], 2 * self.reset_area[3]])
            default_info['visible_id'] = np.where(now_status == 1)[0]
        else:
            old_status = self.memory[-1][2]
            default_info['default_pos'] = self.target_pos_list.copy()
            visible = np.array([], dtype=np.int64)
            for i in range(self.target_num):
                if old_status[i] == 1 and now_status[i] == 0:
                    default_info['default_pos'][i] = self.memory[-1][1][i]
                elif old_status[i] == 0 and now_status[i] == 0:
                    default_info['default_pos'][i] = np.array([2 * self.reset_area[1], 2 * self.reset_area[3]])
                elif old_status[i] == 1 and now_status[i] == 1:
                    visible = np.append(visible, i)
            default_info['visible_id'] = visible
        state = self.predict_model(self.cam, self.target_pos_list, self.memory, 0, default_info)
        return state

    def get_location_from_memory(self, cam_id, memory_id):
        return self.memory[memory_id][0][cam_id]['location'].copy()

    def get_rotation_from_memory(self, cam_id, memory_id):
        return self.memory[memory_id][0][cam_id]['rotation'].copy()

    def get_position_from_memory(self, cam_id, memory_id):
        loc = self.get_location_from_memory(cam_id, memory_id)
        rot = self.get_rotation_from_memory(cam_id, memory_id)
        return loc + rot


class PredictEnv(PoseEnvBase):
    def __init__(self, reset_type, config_path, setting_path, render_save_path, args,
                 cam, target_pos_list, memory, flame, default_info):
        super(PredictEnv, self).__init__(reset_type, config_path, setting_path, render_save_path, args)

        self.default_info = default_info
        self.nav = 'Fix'
        self.load_env_status(cam, target_pos_list)
        self.memory = deepcopy(memory)

        self.action_num = (self.cam_num, self.get_num_actions())

        self.flame = flame
        self.max_predict_flame = self.args.search_depth

        self.now_status = self.get_target_status()
        self.next_pos = self.predict_target_next_pos()

        self.noisy_target_pos_list = self.target_pos_list.copy()

    def predict_target_next_pos(self):
        if self.args.predict_pos:
            if len(self.memory) == 0:
                next_pos = self.default_info['default_pos'].copy()
                next_pos[self.default_info['visible_id']] = self.target_pos_list[self.default_info['visible_id']]
            else:
                visible_next_pos = 2 * self.target_pos_list - self.memory[-1][1]
                visible_next_pos[:, 0] = np.clip(visible_next_pos[:, 0], self.reset_area[0], self.reset_area[1])
                visible_next_pos[:, 1] = np.clip(visible_next_pos[:, 1], self.reset_area[2], self.reset_area[3])
                next_pos = self.default_info['default_pos'].copy()
                next_pos[self.default_info['visible_id']] = visible_next_pos[self.default_info['visible_id']]
        else:
            next_pos = self.default_info['default_pos'].copy()
            next_pos[self.default_info['visible_id']] = self.target_pos_list[self.default_info['visible_id']]

        return next_pos

    def isTerminal(self):
        if self.flame >= self.max_predict_flame:
            return True
        else:
            return False

    def ComputeReward(self):
        reward, team_reward = self.get_reward()
        # return 0.8 * reward + 0.2 * team_reward
        return team_reward

    def takeAction(self, action):
        new_memory = deepcopy(self.memory)
        new_memory.append([deepcopy(self.cam), deepcopy(self.target_pos_list), deepcopy(self.now_status)])
        if len(new_memory) > self.memory_max_len:
            del new_memory[0]

        state = self.predict_model(self.cam, self.next_pos, new_memory, self.flame + 1, self.default_info)

        state.camera_move(action)

        return state
