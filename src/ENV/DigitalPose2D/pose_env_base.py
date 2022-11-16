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

        # noisy init
        # self.noisy_range = self.args.noisy
        # self.noisy_target_pos_list = None
        # print("noisy_data: ", self.noisy_range)

        self.count_steps = 0

        self.memory = []
        self.memory_max_len = 10

    def get_env_info(self):
        env_info = {"state_shape": self.get_state_size(),
                    "obs_shape": self.get_obs_size(),
                    "n_actions": self.get_num_actions(),
                    "n_agents": self.get_num_agents(),
                    "episode_limit": self.get_episode_len(),
                    "n_nodes": self.get_num_nodes(),
                    "node_shape": self.get_node_feature_size(),
                    "edge_shape": self.get_edge_feature_size(),
                    "centralized_obs_shape": self.get_centralized_obs_size()}   # change
        return env_info

    def step(self, actions):
        """A single environment step. 
        actions: a list of size [n_agents], actions[i] is the action taken by the i^th agent
        Returns reward, team_reward, done.
        reward大小为[n_agents]， reward[i]为第i个agent的个人奖励"""
        status = self.get_target_status()
        # self.memory.append([deepcopy(self.cam), deepcopy(self.noisy_target_pos_list), deepcopy(status)])
        self.memory.append([deepcopy(self.cam), deepcopy(self.target_pos_list), deepcopy(status)])
        if len(self.memory) > self.memory_max_len:
            del self.memory[0]

        actions = [int(a) for a in actions]
        # target/camera move
        self.env_one_step(actions)
        # noisy
        # self.get_noisy_pos()

        # reward
        reward, team_reward = self.get_reward()

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
            # reward[i] = sum(cam_target[i]) / self.target_num
        repeat = np.int64(cam_target.sum(axis=0) > 1)
        del_repeat = np.int64(cam_target - repeat > 0)
        for i, cam in enumerate(self.cam_id):
            reward[i] = sum(del_repeat[i]) / self.target_num
        team_reward = np.sum(np.sum(cam_target, 0) > 0) / self.target_num
        # cost = np.sum(np.array(actions) > 0) / self.cam_num
        return reward, team_reward
    
    def reset(self):
        """Reset the environment."""
        self.count_steps = 0

        self.random_reset()
        # noisy
        # self.get_noisy_pos()

        self.memory = []

    # def get_noisy_pos(self):
    #     self.noisy_target_pos_list = (np.random.rand(self.target_num, 2) * 2 - 1) * self.noisy_range +\
    #                                  self.target_pos_list

    def get_memory_obs(self):
        """self.memory[0] is the olddest memory."""
        saved_memory_len = len(self.memory)
        # memory_size = 2 * self.target_num
        memory_obs = np.array([])
        for i in range(self.memory_max_len):
            if i < saved_memory_len:
                memory_obs = np.concatenate([memory_obs, self.memory[i][1].reshape(-1) / self.scale])
            else:
                memory_obs = np.concatenate([np.array([2 * self.reset_area[1], 2 * self.reset_area[3]] * self.target_num) / self.scale, memory_obs])
        return memory_obs

    def get_add_memory_centralized_obs(self):
        """输入就包含历史信息，不做mcts"""
        cobs = []
        for i, cam in enumerate(self.cam_id):
            j_now_visible = []
            obs = []
            cam_pos = self.get_position(cam)
            for j in range(self.target_num):
                angle, dist = self.get_relative_position(cam_pos, self.target_pos_list[j])
                if self.visible(angle, dist):
                    j_now_visible.append(1)
                    obs += [dist / self.scale, np.sin(angle / 180 * np.pi), np.cos(angle / 180 * np.pi)]
                else:
                    j_now_visible.append(0)
                    obs += [-1, -1, -1]
            obs += [(cam_pos[0] - self.mean[0]) / self.scale, (cam_pos[1] - self.mean[1]) / self.scale,
                    np.sin(cam_pos[2] / 180 * np.pi), np.cos(cam_pos[2] / 180 * np.pi)]
            # memory old
            # if len(self.memory) == 0:
            #     cam_pos = self.get_position(cam)
            #     target_pos = self.target_pos_list
            # else:
            #     cam_pos = self.get_position_from_memory(cam, -1)
            #     target_pos = self.memory[-1][1]
            # for j in range(self.target_num):
            #     angle, dist = self.get_relative_position(cam_pos, target_pos[j])
            #     if self.visible(angle, dist):
            #         obs += [dist / self.scale, np.sin(angle / 180 * np.pi), np.cos(angle / 180 * np.pi)]
            #     else:
            #         obs += [-1, -1, -1]
            # obs += [(cam_pos[0] - self.mean[0]) / self.scale, (cam_pos[1] - self.mean[1]) / self.scale,
            #         np.sin(cam_pos[2] / 180 * np.pi), np.cos(cam_pos[2] / 180 * np.pi)]

            # memory future
            if len(self.memory) == 0:
                past_cam_pos = self.get_position(cam)
                past_pos = self.target_pos_list
                future_pos = self.target_pos_list
            else:
                past_cam_pos = self.get_position_from_memory(cam, -1)
                past_pos = self.memory[-1][1]
                future_pos = 2 * self.target_pos_list - self.memory[-1][1]
            for j in range(self.target_num):
                angle, dist = self.get_relative_position(past_cam_pos, past_pos[j])
                if self.visible(angle, dist) or j_now_visible[j] == 1:
                    angle, dist = self.get_relative_position(cam_pos, future_pos[j])
                    obs += [dist / self.scale, np.sin(angle / 180 * np.pi), np.cos(angle / 180 * np.pi)]
                else:
                    obs += [-1, -1, -1]
            # obs += [(cam_pos[0] - self.mean[0]) / self.scale, (cam_pos[1] - self.mean[1]) / self.scale,
            #         np.sin(cam_pos[2] / 180 * np.pi), np.cos(cam_pos[2] / 180 * np.pi)]

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

    def get_add_memory_centralized_obs_size(self):
        obs_size = (self.target_num * 6 + 4 + self.cam_num) * self.cam_num
        # assert len(self.get_add_memory_centralized_obs()[0]) == obs_size
        return obs_size

    def get_normed_target_pos_list(self):
        value = self.target_pos_list.reshape(-1) / self.scale
        return value.copy()

    def get_obs(self):
        """Returns all agent observations in a list.
        [d, sin(theta), cos(theta)]*m (x,y)*n sin(\gamma) cos(\gamma)
        """
        agents_obs = []
        cam_pos = []
        for i, cam in enumerate(self.cam_id):
            now_cam_pos = self.get_position(cam)
            cam_pos.append(now_cam_pos)
        for i, cam in enumerate(self.cam_id):
            obs = []
            for j in range(self.target_num):
                angle, dist = self.get_relative_position(cam_pos[i], self.target_pos_list[j])
                if self.visible(angle, dist):
                    obs += [dist / self.scale, np.sin(angle/180*np.pi), np.cos(angle/180*np.pi)]
                else:
                    obs += [-1, -1, -1]
            for k in range(self.cam_num):
                if k != i:
                    obs += [(cam_pos[k][0] - self.mean[0])/self.scale, (cam_pos[k][1] - self.mean[1])/self.scale]
            obs += [(cam_pos[i][0] - self.mean[0])/self.scale, (cam_pos[i][1] - self.mean[1])/self.scale]
            gamma = self.get_rotation(cam)[0]
            obs += [np.sin(gamma/180*np.pi), np.cos(gamma/180*np.pi)]
            agents_obs.append(np.array(obs))
        return agents_obs

    def get_obs_size(self):
        """Returns the size of the observation."""
        obs_size = 3 * self.target_num + 2 * self.cam_num + 2
        # assert obs_size == len(self.get_obs()[0])
        return obs_size

    def get_new_centralized_obs(self):
        cam_pos_list = []
        obs_list = []
        for i, cam in enumerate(self.cam_id):
            cam_pos = self.get_position(cam)
            cam_pos_list.append(cam_pos)
            obs_list.append([(cam_pos[0] - self.mean[0]) / self.scale, (cam_pos[1] - self.mean[1]) / self.scale,
                             np.sin(cam_pos[2] / 180 * np.pi), np.cos(cam_pos[2] / 180 * np.pi)])

        obs = []
        for j in range(self.target_num):
            find = 0
            for i in range(len(cam_pos_list)):
                angle, dist = self.get_relative_position(cam_pos_list[i], self.target_pos_list[j])
                if self.visible(angle, dist):
                    find = 1
                    obs += [(self.target_pos_list[j][0] - self.mean[0]) / self.scale,
                            (self.target_pos_list[j][1] - self.mean[1]) / self.scale]
                    break
            if find == 0:
                obs += [-1, -1]

        centralized_obs = []
        for i in range(self.cam_num):
            now_obs = obs_list[i].copy()
            for j in range(self.cam_num):
                if j != i:
                    now_obs = np.append(now_obs, obs_list[j])
            now_obs = np.append(now_obs, obs)
            centralized_obs.append(now_obs)
        return centralized_obs

    def get_new_centralized_obs_size(self):
        """Returns the size of the centralized observation."""
        obs_size = self.target_num * 2 + self.cam_num * 4
        # assert len(self.get_new_centralized_obs()[0]) == obs_size
        return obs_size

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

    # def get_noisy_centralized_obs(self):
    #     cobs = []
    #     for i, cam in enumerate(self.cam_id):
    #         obs = []
    #         cam_pos = self.get_position(cam)
    #         for j in range(self.target_num):
    #             angle, dist = self.get_relative_position(cam_pos, self.noisy_target_pos_list[j])
    #             if self.visible(angle, dist):
    #                 obs += [dist / self.scale, np.sin(angle / 180 * np.pi), np.cos(angle / 180 * np.pi)]
    #             else:
    #                 obs += [-1, -1, -1]
    #         obs += [(cam_pos[0] - self.mean[0]) / self.scale, (cam_pos[1] - self.mean[1]) / self.scale,
    #                 np.sin(cam_pos[2] / 180 * np.pi), np.cos(cam_pos[2] / 180 * np.pi)]
    #         one_hot = [0] * self.cam_num
    #         one_hot[i] = 1
    #         obs += one_hot
    #         cobs.append(np.array(obs))
    #
    #     centralized_obs = []
    #     for i in range(self.cam_num):
    #         now_obs = cobs[i].copy()
    #         for j in range(self.cam_num):
    #             if j != i:
    #                 now_obs = np.append(now_obs, cobs[j])
    #         # now_obs = np.append(now_obs, self.count_steps / self.max_steps)
    #         centralized_obs.append(now_obs)
    #     return centralized_obs

    def get_centralized_obs_size(self):
        """Returns the size of the centralized observation."""
        obs_size = (self.target_num * 3 + 4 + self.cam_num) * self.cam_num
        # assert len(self.get_centralized_obs()[0]) == obs_size
        return obs_size

    def get_state(self):
        """Returns the global state.
        [x,y]*m [x,y,sin(\gamma),cos(\gamma)]*n
        """
        state = ((np.array(self.target_pos_list) - np.array(self.mean)) / self.scale).flatten()
        for i, cam in enumerate(self.cam_id):
            pos = self.get_position(cam)
            state = np.append(state, [pos[0] / self.scale, pos[1] / self.scale,
                                      np.sin(pos[2]/180*np.pi), np.cos(pos[2]/180*np.pi)])
        return state

    def get_state_size(self):
        """Returns the size of the global state."""
        state_size = 2 * self.target_num + 4 * self.cam_num
        # assert state_size == len(self.get_state())
        return state_size

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

    def get_num_nodes(self):
        """ Returns the number of nodes (cam_num + target_num)"""
        return self.cam_num + self.target_num

    def get_node_features(self):
        """ Returns all node features (normalized), size: [n_nodes, node_shape]
        target: [0, x, y, -1, -1] * m
        camera: [1, x, y, sin(\gamma), cos(\gamma)] * n
        """
        node_features = np.zeros((self.get_num_nodes(), 5))
        # target features
        norm_target_list = (np.array(self.target_pos_list) - np.array(self.mean)) / self.scale
        node_features[:self.target_num, 1:3] = norm_target_list
        node_features[:self.target_num, 3:] = -1
        # camera features
        node_features[self.target_num:, 0] = 1
        norm_cam_pos = []
        for i, cam in enumerate(self.cam_id):
            now_cam_pos = self.get_position(cam)
            norm_cam_pos.append([(now_cam_pos[0] - self.mean[0])/self.scale, (now_cam_pos[1] - self.mean[1])/self.scale,
                                 np.sin(now_cam_pos[2]/180*np.pi), np.cos(now_cam_pos[2]/180*np.pi)])
        node_features[self.target_num:, 1:] = np.array(norm_cam_pos)
        return node_features

    def get_node_feature_size(self):
        """ Returns the shape of single node feature"""
        node_feature_size = 1 + len(self.get_location(self.cam_id[0])) + 2 * len(self.get_rotation(self.cam_id[0]))
        # assert node_feature_size == self.get_node_features().shape[1]
        return node_feature_size

    def get_adj_matrix(self):
        """ Returns the adjacent matrix between nodes, size: [n_nodes, n_nodes]"""
        num_nodes = self.get_num_nodes()
        adj_matrix = np.zeros((num_nodes, num_nodes))
        for i, cam in enumerate(self.cam_id):
            cam_pos = self.get_position(cam)
            for j in range(self.target_num):
                angle, dist = self.get_relative_position(cam_pos, self.target_pos_list[j])
                if dist <= self.visual_distance:
                    adj_matrix[j, i + self.target_num] = 1
                    adj_matrix[i + self.target_num, j] = 1
        return adj_matrix

    def get_weighted_adj_matrix(self):
        """ Returns the weighted(1-d/r) adjacent matrix between nodes, size: [n_nodes, n_nodes]"""
        num_nodes = self.get_num_nodes()
        weighted_adj_matrix = np.zeros((num_nodes, num_nodes))
        for i, cam in enumerate(self.cam_id):
            cam_pos = self.get_position(cam)
            for j in range(self.target_num):
                angle, dist = self.get_relative_position(cam_pos, self.target_pos_list[j])
                if dist <= self.visual_distance:
                    weighted_adj_matrix[j, i + self.target_num] = 1 - dist / self.visual_distance
                    weighted_adj_matrix[i + self.target_num, j] = 1 - dist / self.visual_distance
        return weighted_adj_matrix

    def get_edge_features(self):
        """ Returns all edge features (normalized),
        i.e. (d, sin(\theta), cos(\theta)), size: [n_nodes, n_nodes, edge_shape]"""
        num_nodes = self.get_num_nodes()
        edge_features = np.zeros((num_nodes, num_nodes, 3))
        for i, cam in enumerate(self.cam_id):
            cam_pos = self.get_position(cam)
            for j in range(self.target_num):
                angle, dist = self.get_relative_position(cam_pos, self.target_pos_list[j])
                if dist <= self.visual_distance:
                    edge_features[j, i + self.target_num] = \
                        np.array([dist / self.scale, np.sin(angle/180*np.pi), np.cos(angle/180*np.pi)])
                    edge_features[i + self.target_num, j] = \
                        np.array([dist / self.scale, np.sin(angle/180*np.pi), np.cos(angle/180*np.pi)])
        return edge_features

    def get_edge_feature_size(self):
        """ Returns the shape of single edge feature"""
        size = 3
        # assert self.get_edge_features().shape[-1] == size
        return size

    def get_predict_model(self):
        default_info = dict()
        now_status = self.get_target_status()
        if len(self.memory) == 0:
            default_info['default_pos'] = self.target_pos_list.copy()
            # default_info['default_pos'] = self.noisy_target_pos_list.copy()
            for i in range(self.target_num):
                default_info['default_pos'][i] = np.array([2 * self.reset_area[1], 2 * self.reset_area[3]])
            default_info['visible_id'] = np.where(now_status == 1)[0]
        else:
            old_status = self.memory[-1][2]
            default_info['default_pos'] = self.target_pos_list.copy()
            # default_info['default_pos'] = self.noisy_target_pos_list.copy()
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
        # state = self.predict_model(self.cam, self.noisy_target_pos_list, self.memory, 0, default_info)
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
        self.nav = 'Fix' #
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
