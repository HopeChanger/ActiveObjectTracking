from __future__ import division

import time
import random
import numpy as np
import matplotlib.pyplot as plt
import os


layer_dis = 10
node_size = 2


def paint_children(node, pos, width):
    keys = list(node.children.keys())
    children_num = len(keys)
    if children_num == 0:
        return
    elif children_num == 1:
        child_pos = (pos[0], pos[1] - layer_dis)
        plt.gca().add_artist(plt.Circle(child_pos, node_size, color='green', fill=True,
                                        alpha=np.clip(np.sum(node.numVisits) / 60, 0, 1)))
        plt.annotate(str(int(np.sum(node.numVisits) / 6)), xy=child_pos, xytext=child_pos, fontsize=10, color='maroon')
        plt.plot([pos[0], child_pos[0]], [pos[1], child_pos[1]], color='black')
        paint_children(node.children[keys[0]], child_pos, width * 0.9)
    else:
        for i in range(children_num):
            child_pos = (pos[0]-width/2+width*i/(children_num-1), pos[1]-layer_dis)
            plt.gca().add_artist(plt.Circle(child_pos, node_size, color='green', fill=True,
                                            alpha=np.clip(np.sum(node.numVisits) / 60, 0, 1)))
            plt.annotate(str(int(np.sum(node.numVisits) / 6)), xy=child_pos, xytext=child_pos, fontsize=10, color='maroon')
            plt.plot([pos[0], child_pos[0]], [pos[1], child_pos[1]], color='black')
            paint_children(node.children[keys[i]], child_pos, width * 0.9 / children_num)


def render(root, save_path):
    plt.figure()
    plt.cla()
    plt.axis("equal")
    # plt.axis([-10, 10, -100, 0])
    root_pos = (0, 0)
    plt.gca().add_artist(plt.Circle(root_pos, node_size, color='green', fill=True,
                                    alpha=np.clip(np.sum(root.numVisits) / 60, 0, 1)))
    plt.annotate(str(int(np.sum(root.numVisits) / 6)), xy=root_pos, xytext=root_pos, fontsize=10, color='maroon')
    paint_children(root, root_pos, 100)
    plt.axis('off')
    plt.savefig(save_path)
    # plt.show()


class TreeNode(object):
    def __init__(self, state, parent, net, hidden_states=None):
        self.state = state
        self.isTerminal = state.isTerminal()
        self.reward = state.ComputeReward()

        self.N = np.zeros(state.action_num[0] * (state.action_num[1] - 1) + 1)
        self.init_V = np.ones(state.action_num[0] * (state.action_num[1] - 1) + 1)

        raw_q_value, self.hidden = net(state, hidden_states)
        self.best_action = np.argmax(raw_q_value, axis=1)

        for i in range(state.action_num[0]):
            for j in range(state.action_num[1] - 1):
                if j < self.best_action[i]:
                    self.init_V[i * (state.action_num[1] - 1) + j] = raw_q_value[i, j] / raw_q_value[i, self.best_action[i]]
                else:
                    self.init_V[i * (state.action_num[1] - 1) + j] = raw_q_value[i, j + 1] / raw_q_value[i, self.best_action[i]]

        self.V = self.init_V.copy()

        self.action_shape = state.action_num

        self.parent = parent
        self.children = {}

    def __str__(self):
        s = []
        s.append("isTerminal: %s" % self.isTerminal)
        s.append("reward: %s" % self.reward)
        s.append("best_action: %s" % self.best_action)
        s.append("N: %s" % self.N)
        s.append("init_V: %s" % self.init_V)
        s.append("V: %s" % self.V)
        return "%s: {%s}" % (self.__class__.__name__, ', '.join(s))


class MCTS(object):
    def __init__(self, search_times, eta=1.0, net=None):
        assert search_times > 0

        self.search_times = search_times
        self.eta = eta
        self.net = net

        self.root = None
        self.step_child_list = []

    def search(self, init_state, init_hidden, info=None):
        self.root = TreeNode(init_state, None, self.net, init_hidden)

        for i in range(self.search_times):
            self.step_child_list = []
            node = self.tree_policy()
            rollout_reward, depth = self.default_policy(node)
            self.backup(node, rollout_reward, depth)

        action_id = np.argmax(self.root.N + self.root.V / 100)
        # action_id = random.randint(0, 49)
        action = self.id2action(self.root, action_id)
        return [action]

    def id2action(self, node, child_id):
        best_action = node.best_action.copy()
        if child_id < node.action_shape[0] * (node.action_shape[1] - 1):
            change_cam = child_id // (node.action_shape[1] - 1)
            change_id = child_id % (node.action_shape[1] - 1)
            if change_id >= best_action[change_cam]:
                best_action[change_cam] = change_id + 1
            else:
                best_action[change_cam] = change_id
        return best_action

    def tree_policy(self):
        node = self.root
        while not node.isTerminal:
            child_id = self.choose_best_child(node)
            self.step_child_list.append(child_id)
            if child_id in node.children:
                node = node.children[child_id]
            else:
                action = self.id2action(node, child_id)
                node.children[child_id] = TreeNode(node.state.takeAction(action), node, self.net, node.hidden)
                return node.children[child_id]
        return node

    def choose_best_child(self, node):
        # epsilon = 1e-5
        # score = node.V + np.sqrt(2 * np.log(np.sum(node.N)) / (node.N + epsilon))
        # return np.argmax(score)
        return np.argmax(node.V)

    def default_policy(self, node):
        now_state = node.state
        rollout_reward = 0
        depth = 0
        hidden = node.hidden
        while not now_state.isTerminal():
            raw_q_value, hidden = self.net(now_state, hidden)
            best_action = np.argmax(raw_q_value, axis=1)
            now_state = now_state.takeAction(best_action)
            rollout_reward += now_state.ComputeReward()
            depth += 1
        return rollout_reward, depth

    def backup(self, node, rollout_reward, depth):
        cluster = -1
        reward_sum = rollout_reward
        while node.parent is not None:
            reward_sum += node.reward
            depth += 1
            node = node.parent
            child_id = self.step_child_list[cluster]
            node.N[child_id] += 1
            # node.V[child_id] = node.V[child_id] + (reward_sum / depth - node.V[child_id]) / node.N[child_id]  # non-init
            node.V[child_id] = node.V[child_id] + (reward_sum / depth - node.V[child_id]) / (node.N[child_id] + 1) # init
            cluster -= 1
