# from envs import REGISTRY as env_REGISTRY
from ENV.DigitalPose2D.pose_env_base import PoseEnvBase
from functools import partial
from components.episode_buffer import EpisodeBatch
import numpy as np
from mcts.mcts import MCTS
import torch as th


def team_spirit(rew, team_rew, tau):
    rew = np.array(rew)
    return (1 - tau) * rew + tau * team_rew


class EpisodeRunner:

    def __init__(self, args, logger):
        self.args = args
        self.logger = logger
        self.batch_size = self.args.batch_size_run
        assert self.batch_size == 1

        # self.env = env_REGISTRY[self.args.env](**self.args.env_args)
        self.env = PoseEnvBase(config_path=self.args.config_path, render_save_path=args.render_save_path, args=self.args)
        self.env_info = self.env.get_env_info()
        self.episode_limit = self.env_info['episode_limit']
        self.t = 0

        self.t_env = 0

        self.train_returns = []
        self.test_returns = []
        self.train_stats = {}
        self.test_stats = {}

        # Log the first run
        self.log_train_stats_t = -1000000

        if self.args.evaluate_with_mcts:
            self.mcts = MCTS(search_times=self.args.mcts_times, eta=1.0, net=self.state_to_network_output)

    def setup(self, scheme, groups, preprocess, mac):
        self.new_batch = partial(EpisodeBatch, scheme, groups, self.batch_size, self.episode_limit + 1,
                                 preprocess=preprocess, device=self.args.device)
        self.mac = mac

    def get_env_info(self):
        return self.env.get_env_info()

    def reset(self):
        self.batch = self.new_batch()
        self.env.reset()
        self.t = 0
        if self.args.evaluate_with_mcts:
            self.mcts_batch = self.new_batch()

    def state_to_network_output(self, state, hidden_states):
        input_data = {
            "avail_actions": [state.get_avail_actions()],
            "obs": [state.get_centralized_obs()]
        }
        self.mcts_batch.update(input_data, ts=0)
        self.mac.load_hidden(hidden_states)
        with th.no_grad():
            output = self.mac.forward(self.mcts_batch, 0, test_mode=True)
            output = output[0]
        output_hidden = self.mac.read_hidden()
        return output.cpu().numpy(), output_hidden

    def run(self, test_mode=False):
        self.reset()

        terminated = False
        episode_return = 0
        self.mac.init_hidden(batch_size=self.batch_size)

        while not terminated:

            pre_transition_data = {
                "avail_actions": [self.env.get_avail_actions()],
                "obs": [self.env.get_centralized_obs()]
            }
            
            self.batch.update(pre_transition_data, ts=self.t)

            if self.args.evaluate_with_mcts:
                save_hidden = self.mac.read_hidden()
                actions = self.mcts.search(self.env.get_predict_model(), save_hidden, {'n': self.t})
                if self.args.log_file_name != "":
                    with open(self.args.log_file_name, 'a') as f:
                        f.write(str(self.mcts.root))
                        f.write('\n')
                self.mac.load_hidden(save_hidden)
                print("a^mcts: {}".format(actions))

                old_action = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode)
                print("a^net: {}".format(old_action))
            else:
                actions = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode)
                # actions = [np.random.randint(0, 9, 4)]  # random

            reward, team_reward, terminated = self.env.step(actions[0])
            
            episode_return += team_reward

            post_transition_data = {
                "actions": actions,
                "reward": [team_spirit(reward, team_reward, self.args.tau)],
                "terminated": [(terminated,)],
            }

            self.batch.update(post_transition_data, ts=self.t)

            self.t += 1

        last_data = {
            "avail_actions": [self.env.get_avail_actions()],
            "obs": [self.env.get_centralized_obs()]
        }

        self.batch.update(last_data, ts=self.t)

        # Select actions in the last stored state
        actions = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode)
        self.batch.update({"actions": actions}, ts=self.t)

        cur_stats = self.test_stats if test_mode else self.train_stats
        cur_returns = self.test_returns if test_mode else self.train_returns
        log_prefix = "test_" if test_mode else ""
        cur_stats["n_episodes"] = 1 + cur_stats.get("n_episodes", 0)
        cur_stats["ep_length"] = self.t + cur_stats.get("ep_length", 0)
        cur_stats["ep_rew_mean"] = episode_return / self.t + cur_stats.get("ep_rew_mean", 0)

        if not test_mode:
            self.t_env += self.t

        cur_returns.append(episode_return)

        if self.args.evaluate_with_mcts:
            print("ep_rew_mean: ", cur_returns)
            if self.args.log_file_name != "":
                with open(self.args.log_file_name, 'a') as f:
                    f.write('\n\n')

        if test_mode and (len(self.test_returns) == self.args.test_nepisode):
            self._log(cur_returns, cur_stats, log_prefix)
        elif self.t_env - self.log_train_stats_t >= self.args.runner_log_interval:
            self._log(cur_returns, cur_stats, log_prefix)
            if hasattr(self.mac.action_selector, "epsilon"):
                self.logger.log_stat("epsilon", self.mac.action_selector.epsilon, self.t_env)
            self.log_train_stats_t = self.t_env

        return self.batch

    def _log(self, returns, stats, prefix):
        self.logger.log_stat(prefix + "return_mean", np.mean(returns), self.t_env)
        self.logger.log_stat(prefix + "return_std", np.std(returns), self.t_env)
        returns.clear()

        for k, v in stats.items():
            if k != "n_episodes":
                self.logger.log_stat(prefix + k + "_mean", v/stats["n_episodes"], self.t_env)
        stats.clear()

