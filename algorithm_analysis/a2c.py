# From http://inoryy.com/post/tensorflow2-deep-reinforcement-learning/
import logging
import time
import numpy as np
from tqdm.auto import tqdm
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.losses import mean_squared_error, SparseCategoricalCrossentropy, categorical_crossentropy
from tensorflow.keras.optimizers import RMSprop
from algorithm_analysis.metrics import A2C_MetricByBatch, A2C_MetricByEpisodes

class ProbabilityDistribution(tf.keras.Model):
    def call(self, logits, **kwargs):
        # Sample a random categorical action from the given logits.
        return tf.squeeze(tf.random.categorical(logits, 1), axis=-1)


class Model(tf.keras.Model):
    def __init__(self, num_actions, units=128):
        super().__init__('mlp_policy')
        # Note: no tf.get_variable(), just simple Keras API!
        self.hidden11 = Dense(128, activation='relu')
        #self.hidden12 = Dense(64, activation='relu')
        #self.hidden13 = Dense(28, activation='relu')
        self.hidden21 = Dense(128, activation='relu')
        #self.hidden22 = Dense(64, activation='relu')
        #self.hidden23 = Dense(28, activation='relu')
        self.value = Dense(1, name='value')
        # Logits are unnormalized log probabilities.
        self.logits = Dense(num_actions, name='policy_logits')
        self.dist = ProbabilityDistribution()

    def call(self, inputs, **kwargs):
        # Inputs is a numpy array, convert to a tensor.
        x = tf.convert_to_tensor(inputs)
        # Separate hidden layers from the same input tensor.
        hidden_logs = self.hidden11(x)
        #hidden_logs = self.hidden12(hidden_logs)
        #hidden_logs = self.hidden13(hidden_logs)
        hidden_vals = self.hidden21(x)
        #hidden_vals = self.hidden22(hidden_vals)
        #hidden_vals = self.hidden23(hidden_vals)
        return self.logits(hidden_logs), self.value(hidden_vals)

    def action_value(self, obs):
        # Executes `call()` under the hood.
        logits, value = self.predict_on_batch(obs)
        action = self.dist.predict_on_batch(logits)
        # Another way to sample actions:
        #   action = tf.random.categorical(logits, 1)
        # Will become clearer later why we don't use it.
        return np.squeeze(action, axis=-1), np.squeeze(value, axis=-1)


class A2CAgent:
    def __init__(self, model, learning_rate=7e-3, discount_rate=0.99, value_coefficient=0.5, entropy_coefficient=1e-4):
        self.learning_rate = learning_rate
        self.discount_rate = discount_rate
        self.value_coefficient = value_coefficient
        self.entropy_coefficient = entropy_coefficient

        self.model = model
        self.model.compile(
            optimizer=RMSprop(lr=self.learning_rate),
            # Define separate losses for policy logits and value estimate.
            loss=[self._logits_loss, self._value_loss])

    def train_by_batches(self, env, batch_sz=64, updates=250):
        # Storage helpers for a single batch of data.
        actions = np.empty((batch_sz,), dtype=np.int32)
        rewards, dones, values = np.empty((3, batch_sz))
        observations = np.empty((batch_sz,) + np.array(env.masks[0]).shape)  # !!! change for our broken component env
        # Training loop: collect samples, send to optimizer, repeat updates times.
        ep_rewards = [0.0]
        ep_lengths = [0]
        next_obs = env.reset()
        for update in tqdm(range(updates), desc='Update Loop: '):
            for step in range(batch_sz):
                observations[step] = next_obs.copy()
                actions[step], values[step] = self.model.action_value(next_obs[None, :])
                next_obs, rewards[step], dones[step], _ = env.step(actions[step])

                ep_rewards[-1] += rewards[step]
                ep_lengths[-1] += 1
                if dones[step]:
                    ep_rewards.append(0.0)
                    ep_lengths.append(0)
                    next_obs = env.reset()
                    logging.info("Episode: %03d, Reward: %03d" % (len(ep_rewards) - 1, ep_rewards[-2]))

            _, next_value = self.model.action_value(next_obs[None, :])
            returns, advs = self._returns_advantages(rewards, dones, values, next_value)
            # A trick to input actions and advantages through same API.
            acts_and_advs = np.concatenate([actions[:, None], advs[:, None]], axis=-1)
            # Performs a full training step on the collected batch.
            # Note: no need to mess around with gradients, Keras API handles it.
            losses = self.model.train_on_batch(observations, [acts_and_advs, returns])
            logging.debug("[%d/%d] Losses: %s" % (update + 1, updates, losses))

        metrics = A2C_MetricByBatch(batch_size=batch_sz, 
                                updates=updates, 
                                episode_lengths=ep_lengths, 
                                rewards=ep_rewards,
                                learning_rate=self.learning_rate,
                                discount_rate=self.discount_rate,
                                value_coefficient=self.value_coefficient,
                                entropy_coefficient=self.entropy_coefficient)
        return metrics

    
    def train_by_episodes(self, env, episodes=250):
        ep_rewards = [0.0]
        ep_lengths = [0]
        ep_skips = []
        step = 0
        for episode in tqdm(range(episodes), desc='Episode Loop: '):
            actions = np.empty((1,), dtype=np.int32)
            rewards, dones, values = np.empty((3, 1))
            observations = np.empty((1,) + np.array(env.masks[0]).shape)  # !!! change for our broken component env
            next_obs = env.reset()
            dones[step] = 0
            ep_step = 0
            #step_thresh = env.action_space.n*2
            while not dones[step]:
                observations[step] = next_obs.copy()
                actions[step], values[step] = self.model.action_value(next_obs[None, :])
                next_obs, rewards[step], dones[step], _ = env.step(actions[step])
                
                ep_rewards[-1] += rewards[step]
                ep_lengths[-1] += 1

                _, next_value = self.model.action_value(next_obs[None, :])
                returns, advs = self._returns_advantages(rewards, dones, values, next_value)
                # A trick to input actions and advantages through same API.
                acts_and_advs = np.concatenate([actions[:, None], advs[:, None]], axis=-1)
                # Performs a full training step on the collected batch.
                # Note: no need to mess around with gradients, Keras API handles it.
                losses = self.model.train_on_batch(observations, [acts_and_advs, returns])
                logging.debug("[%d/%d] Losses: %s" % (episode + 1, episodes, losses))
                
                # clear_output(wait=True)
                # print(episode, ep_step)
                
                if dones[step] and (episode < episodes-1):
                    ep_rewards.append(0.0)
                    ep_lengths.append(0)
                    logging.info("Episode: %03d, Reward: %03d" % (len(ep_rewards) - 1, ep_rewards[-2]))
                    
                #if ep_lengths[-1] > step_thresh:
                #    ep_rewards.append(0.0)
                #    ep_lengths.append(0)
                #    ep_skips.append(episode)
                #    break
        
        metrics = A2C_MetricByEpisodes(episodes=episodes,
                                       episode_lengths=ep_lengths,
                                       rewards=ep_rewards,
                                       learning_rate=self.learning_rate,
                                       discount_rate=self.discount_rate,
                                       value_coefficient=self.value_coefficient,
                                       entropy_coefficient=self.entropy_coefficient,
                                       num_broken_components=env.action_space.n,
                                       episode_skips=ep_skips
                                      )
        return metrics
    
    
    def train_by_episodes_batches(self, env, episodes=500, batch_sz=64, updates=250):
        # Storage helpers for a single batch of data.
        actions = np.empty((batch_sz,), dtype=np.int32)
        rewards, dones, values = np.empty((3, batch_sz))
        observations = np.empty((batch_sz,) + np.array(env.masks[0]).shape)  # !!! change for our broken component env
        # Training loop: collect samples, send to optimizer, repeat updates times.
        ep_rewards = [0.0]
        ep_lengths = [0]
        next_obs = env.reset()
        done_counter = 0
        while len(ep_rewards) <= episodes:
            for update in tqdm(range(updates), desc='Episode: {}; Update Loop: '.format(len(ep_rewards))):
                for step in range(batch_sz):
                    observations[step] = next_obs.copy()
                    actions[step], values[step] = self.model.action_value(next_obs[None, :])
                    next_obs, rewards[step], dones[step], _ = env.step(actions[step])

                    ep_rewards[-1] += rewards[step]
                    ep_lengths[-1] += 1
                    if dones[step]:
                        ep_rewards.append(0.0)
                        ep_lengths.append(0)
                        next_obs = env.reset()
                        logging.info("Episode: %03d, Reward: %03d" % (len(ep_rewards) - 1, ep_rewards[-2]))

                _, next_value = self.model.action_value(next_obs[None, :])
                returns, advs = self._returns_advantages(rewards, dones, values, next_value)
                # A trick to input actions and advantages through same API.
                acts_and_advs = np.concatenate([actions[:, None], advs[:, None]], axis=-1)
                # Performs a full training step on the collected batch.
                # Note: no need to mess around with gradients, Keras API handles it.
                losses = self.model.train_on_batch(observations, [acts_and_advs, returns])
                logging.debug("[%d/%d] Losses: %s" % (update + 1, updates, losses))

        metrics = A2C_MetricByBatch(batch_size=batch_sz, 
                                updates=updates, 
                                episode_lengths=ep_lengths, 
                                rewards=ep_rewards,
                                learning_rate=self.learning_rate,
                                discount_rate=self.discount_rate,
                                value_coefficient=self.value_coefficient,
                                entropy_coefficient=self.entropy_coefficient,
                                num_broken_components=env.action_space.n
                                   )
        return metrics
    

    def test(self, env, render=False):
        obs, done, ep_reward = env.reset(), False, 0
        while not done:
            action, _ = self.model.action_value(obs[None, :])
            obs, reward, done, _ = env.step(action)
            ep_reward += reward
            if render:
                env.render()
        return ep_reward

    def _returns_advantages(self, rewards, dones, values, next_value):
        # `next_value` is the bootstrap value estimate of the future state (critic).
        returns = np.append(np.zeros_like(rewards), next_value, axis=-1)
        # Returns are calculated as discounted sum of future rewards.
        for t in reversed(range(rewards.shape[0])):
            returns[t] = rewards[t] + self.discount_rate * returns[t + 1] * (1 - dones[t])
        returns = returns[:-1]
        # Advantages are equal to returns - baseline (value estimates in our case).
        advantages = returns - values
        return returns, advantages

    def _value_loss(self, returns, value):
        # Value loss is typically MSE between value estimates and returns.
        return self.value_coefficient * mean_squared_error(returns, value)

    def _logits_loss(self, actions_and_advantages, logits):
        # A trick to input actions and advantages through the same API.
        actions, advantages = tf.split(actions_and_advantages, 2, axis=-1)
        # Sparse categorical CE loss obj that supports sample_weight arg on `call()`.
        # `from_logits` argument ensures transformation into normalized probabilities.
        weighted_sparse_ce = SparseCategoricalCrossentropy(from_logits=True)
        # Policy loss is defined by policy gradients, weighted by advantages.
        # Note: we only calculate the loss on the actions we've actually taken.
        actions = tf.cast(actions, tf.int32)
        policy_loss = weighted_sparse_ce(actions, logits, sample_weight=advantages)
        # Entropy loss can be calculated as cross-entropy over itself.
        probs = tf.nn.softmax(logits)
        entropy_loss = categorical_crossentropy(probs, probs)
        # We want to minimize policy and maximize entropy losses.
        # Here signs are flipped because the optimizer minimizes.
        return policy_loss - self.entropy_coefficient * entropy_loss