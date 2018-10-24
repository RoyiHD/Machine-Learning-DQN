import tensorflow as tf
import os
import sys
import numpy as np
import gym
import itertools
from collections import deque, namedtuple
import random
from gym.wrappers import Monitor


"DEPENDENCY ffmpeg via brew"


env = gym.make('Breakout-v0')



VALID_ACTIONS = [0,1,2,3]






class StateProcessor:
    def __init__(self):
        with tf.variable_scope('state_processor'):
            self.input_state = tf.placeholder(shape=[210,160,3], dtype=tf.uint8)
            self.output = tf.image.rgb_to_grayscale(self.input_state)
            self.output = tf.image.crop_to_bounding_box(self.output, 34, 0, 160, 160)
            self.output = tf.image.resize_images(
                self.output, [84, 84], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            self.output = tf.squeeze(self.output)


    def process(self, sess, state):
        """

        :param sess:
        :param state: [210,160,3] Atari RGB state
        :return: [84,84,1] grayscale state
        """
        return sess.run(self.output, {self.input_state:state})


class Estimator:

    def __init__(self, scope='estimator', summaries_dir=None):
        self.scope = scope
        self.summary_writer = None
        with tf.variable_scope(scope):
            self.build_model()
            if summaries_dir:
                summary_dir = os.path.join(summaries_dir, "summaries_{}".format(scope))
                if not os.path.exists(summary_dir):
                    os.makedirs(summary_dir)
                self.summary_writer = tf.summary.FileWriter(summary_dir)


    def build_model(self):

        self.X = tf.placeholder(shape=[None, 84,84,4], dtype=tf.uint8, name='X')
        self.y_pl = tf.placeholder(shape=[None], dtype=tf.float32, name="y")
        # Integer id of which action was selected
        self.actions_pl = tf.placeholder(shape=[None], dtype=tf.int32, name="actions")

        X = tf.to_float(self.X) / 255.0
        batch_size = tf.shape(self.X)[0]

        conv1 = tf.contrib.layers.conv2d(X, 32, 8, 4, activation_fn=tf.nn.relu)
        conv2 = tf.contrib.layers.conv2d(conv1, 64, 4, 2, activation_fn=tf.nn.relu)
        conv3 = tf.contrib.layers.conv2d(conv2, 64, 3, 1, activation_fn=tf.nn.relu)

        flat = tf.contrib.layers.flatten(conv3)
        fc1 = tf.contrib.layers.fully_connected(flat, 512)
        self.predictions = tf.contrib.layers.fully_connected(fc1, len(VALID_ACTIONS))

        gather_indices = tf.range(batch_size) * tf.shape(self.predictions)[1] + self.actions_pl
        self.action_predictions = tf.gather(tf.reshape(self.predictions, [-1]), gather_indices)

        self.losses = tf.squared_difference(self.y_pl, self.action_predictions)
        self.loss = tf.reduce_mean(self.losses)


        self.optimizer = tf.train.RMSPropOptimizer(0.00025, 0.99, 0.0, 1e-6)
        self.train_op = self.optimizer.minimize(self.loss, global_step=tf.contrib.framework.get_global_step())

        # Summaries for Tensorboard
        self.summaries = tf.summary.merge([
            tf.summary.scalar("loss", self.loss),
            tf.summary.histogram("loss_hist", self.losses),
            tf.summary.histogram("q_values_hist", self.predictions),
            tf.summary.scalar("max_q_value", tf.reduce_max(self.predictions))
        ])


    def predict(self, sess, d):

        return sess.run(self.predictions, {self.X:d})

    def update(self, sess, s, a, y):
        feed_dict = {self.X:s, self.actions_pl:a, self.y_pl:y}
        summaries, global_step, _, loss = sess.run(
            [self.summaries, tf.contrib.framework.get_global_step(), self.train_op, self.loss],
            feed_dict)
        if self.summary_writer:
            self.summary_writer.add_summary(summaries, global_step)
        return loss



def copy_model_params(sess, estimator1, estimator2):
    e1_params = [t for t in tf.trainable_variables() if t.name.startswith(estimator1.scope)]
    e1_params = sorted(e1_params, key=lambda v: v.name)
    e2_params = [t for t in tf.trainable_variables() if t.name.startswith(estimator2.scope)]
    e2_params = sorted(e2_params, key=lambda v: v.name)
    update_ops = []
    for e1_v, e2_v in zip(e1_params, e2_params):
        op = e2_v.assign(e1_v)
        update_ops.append(op)
    sess.run(update_ops)


def epsilon_greedy(estimator, na):
    def policy_fn(sess, observation, epsilon):
        A = np.ones(na, dtype=float)*epsilon / na
        q_values = estimator.predict(sess, np.expand_dims(observation, 0))[0]
        best_action = np.argmax(q_values)
        A[best_action] += (1.0-epsilon)
        return A
    return policy_fn



def learning(
        sess,
        env,
        q_estimator,
        target_estimator,
        state_processor,
        num_episodes,
        experiment_dir,
        replay_mem=50000,
        target_estimator_update=10000,
        discount=0.99,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay_steps=500000,
        batch=32,
        record_vid=50):

    Transition = namedtuple("Transition", ['state','action','reward','next_state','done'])
    replay_memory = []
    stats = None
    #stats = plotting.EpisodeStats(
     #   episode_lengths=np.zeros(num_episodes),
      #  episode_rewards=np.zeros(num_episodes))

    checkpoint_dir = os.path.join(experiment_dir,'checkpoints')
    checkpoint_path = os.path.join(checkpoint_dir, "model")
    monitor_path = os.path.join(experiment_dir, "monitor")

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    if not os.path.exists(monitor_path):
        os.makedirs(monitor_path)

    saver = tf.train.Saver()

    latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
    if latest_checkpoint:
        print("Loading model checkpoint {}...\n".format(latest_checkpoint))
        saver.restore(sess, latest_checkpoint)


    total_t = sess.run(tf.contrib.framework.get_global_step())
    epsilons = np.linspace(epsilon_start, epsilon_end, epsilon_decay_steps)

    policy = epsilon_greedy(q_estimator, len(VALID_ACTIONS))


    state = env.reset()
    print("shape of state ", state.shape)
    state = state_processor.process(sess, state)
    state = np.stack([state]*4, axis=2)

    for i in range(replay_mem):
        action_probs = policy(sess, state, epsilons[min(total_t, epsilon_decay_steps-1)])
        action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
        next_state, reward, done, _ = env.step(VALID_ACTIONS[action])
        next_state = state_processor.process(sess, next_state)
        next_state = np.append(state[:,:,1:], np.expand_dims(next_state, 2), axis=2)
        replay_memory.append(Transition(state, action, reward, next_state, done))
        if done:
            state = env.reset()
            state = state_processor.process(sess, state)
            state = np.stack([state] * 4, axis=2)
        else:
            state = next_state
        if i % 1000 == 0:
            print("LEN OF MEM ", len(replay_memory))

    #env = Monitor(env, directory=monitor_path, video_callable=lambda count: count % record_vid == 0,
                  #resume=True)
    print("Done populating memory play")
    for i_episode in range(num_episodes):
        # Save the current checkpoint
        saver.save(tf.get_default_session(), checkpoint_path)

        state = env.reset()
        #env.render()
        state = state_processor.process(sess, state)
        state = np.stack([state] * 4, axis=2)
        loss = None

        for t in itertools.count():
            # Epsilon for this time step
            epsilon = epsilons[min(total_t, epsilon_decay_steps - 1)]
            # Add epsilon to Tensorboard
            episode_summary = tf.Summary()
            episode_summary.value.add(simple_value=epsilon, tag="epsilon")
            q_estimator.summary_writer.add_summary(episode_summary, total_t)
            # Maybe update the target estimator
            if total_t % target_estimator_update == 0:
                copy_model_params(sess, q_estimator, target_estimator)
                print("\nCopied model parameters to target network.")
            # Print out which step we're on, useful for debugging.
            print("\rStep {} ({}) @ Episode {}/{}, loss: {}".format(
                t, total_t, i_episode + 1, num_episodes, loss), end="")
            sys.stdout.flush()
            # Take a step
            action_probs = policy(sess, state, epsilon)
            action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
            next_state, reward, done, _ = env.step(VALID_ACTIONS[action])
            next_state = state_processor.process(sess, next_state)
            next_state = np.append(state[:, :, 1:], np.expand_dims(next_state, 2), axis=2)
            # If our replay memory is full, pop the first element
            if len(replay_memory) == replay_mem:
                replay_memory.pop(0)
            # Save transition to replay memory
            replay_memory.append(Transition(state, action, reward, next_state, done))
            # Update statistics
            #stats.episode_rewards[i_episode] += reward
            #stats.episode_lengths[i_episode] = t
            # Sample a minibatch from the replay memory
            samples = random.sample(replay_memory, batch)
            states_batch, action_batch, reward_batch, next_states_batch, done_batch = map(np.array, zip(*samples))
            # Calculate q values and targets
            q_values_next = target_estimator.predict(sess, next_states_batch)
            targets_batch = reward_batch + np.invert(done_batch).astype(np.float32) * discount * np.amax(
                q_values_next, axis=1)
            # Perform gradient descent update
            states_batch = np.array(states_batch)
            loss = q_estimator.update(sess, states_batch, action_batch, targets_batch)
            if done:
                break
            state = next_state
            total_t += 1
            # Add summaries to tensorboard
        episode_summary = tf.Summary()
        #episode_summary.value.add(simple_value=stats.episode_rewards[i_episode], node_name="episode_reward",
         #                         tag="episode_reward")
        #episode_summary.value.add(simple_value=stats.episode_lengths[i_episode], node_name="episode_length",
         #                         tag="episode_length")
        q_estimator.summary_writer.add_summary(episode_summary, total_t)
        q_estimator.summary_writer.flush()
        #yield total_t, plotting.EpisodeStats(
         #   episode_lengths=stats.episode_lengths[:i_episode + 1],
          #  episode_rewards=stats.episode_rewards[:i_episode + 1])
    return stats

def play_game():

    


if __name__ == "__main__":
    tf.reset_default_graph()
    # Where we save our checkpoints and graphs
    experiment_dir = os.path.abspath("./experiments/{}".format(env.spec.id))
    # Create a glboal step variable
    global_step = tf.Variable(0, name='global_step', trainable=False)

    # Create estimators
    q_estimator = Estimator(scope="q", summaries_dir=experiment_dir)
    target_estimator = Estimator(scope="target_q")
    # State processor
    state_processor = StateProcessor()
    # Run it!
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for t, stats in learning(sess,
                                        env,
                                        q_estimator=q_estimator,
                                        target_estimator=target_estimator,
                                        state_processor=state_processor,
                                        experiment_dir=experiment_dir,
                                        num_episodes=10000,
                                        replay_mem=5000,
                                        target_estimator_update=10000,
                                        epsilon_start=1.0,
                                        epsilon_end=0.1,
                                        epsilon_decay_steps=500000,
                                        discount=0.99,
                                        batch=32):
            print("\nEpisode Reward: {}".format(stats.episode_rewards[-1]))