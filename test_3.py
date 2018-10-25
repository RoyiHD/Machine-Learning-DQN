import tensorflow as tf
import os
import numpy as np
import gym
from collections import deque
import time

env = gym.make('BreakoutDeterministic-v4')
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


def epsilon_greedy(estimator, na):
    def policy_fn(sess, observation, epsilon):
        A = np.ones(na, dtype=float)*epsilon / na

        q_values = estimator.predict(sess, np.expand_dims(observation, 0))[0]

        best_action = np.argmax(q_values)

        #Strengthening the predicted q when epsilon decreases towards 0
        A[best_action] += (1.0-epsilon)
        return A
    return policy_fn


class Model:

    def __init__(self, batch, scope='estimator'):
        self.scope = scope
        self.batch = batch
        with tf.variable_scope(scope):
            self.global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')
            self.build_model()

    def build_model(self):
        self.X = tf.placeholder(shape=[None,84,84,4], dtype=tf.uint8, name='X')
        #self.actions = tf.placeholder(shape=[None], dtype=tf.int32, name='actions')
        self.actions = tf.placeholder(shape=[None], dtype=tf.int32, name='actions')
        self.rewards = tf.placeholder(shape=[None], dtype=tf.float32, name='rewards')

        x = tf.to_float(self.X) / 255

        #PARAMS:
        # 1st = Input (Img), 2nd = Filters, 3rd = Kernel, 4th = stride
        conv1 = tf.contrib.layers.conv2d(x, 32, [8,8], 4,activation_fn=tf.nn.relu)

        #MAX POOLING
        #pool1 = tf.contrib.layers.max_pool2d(conv1, [2,2], stride=2)

        conv2 = tf.contrib.layers.conv2d(conv1, 64, [4,4], 2, activation_fn=tf.nn.relu)
        conv3 = tf.contrib.layers.conv2d(conv2, 64, [3,3], 1, activation_fn=tf.nn.relu)
        flat = tf.contrib.layers.flatten(conv3)
        fc1 = tf.contrib.layers.fully_connected(flat, 512)

        #DROPOUT
        #dropout = tf.layers.dropout(
         #   inputs=fc1, rate=0.2)

        self.predictions = tf.contrib.layers.fully_connected(fc1, len(VALID_ACTIONS))

        #Select the values from the prediction which match the actions index array
        self.gather_indices = tf.range(self.batch) * tf.shape(self.predictions)[1] + self.actions
        self.action_predictions = tf.gather(tf.reshape(self.predictions, [-1]), self.gather_indices)

        self.loss = tf.reduce_mean(tf.squared_difference(self.rewards, self.action_predictions))

        self.optimizer = tf.train.RMSPropOptimizer(0.001, 0.99, 0.0, 1e-6)
        self.train_op = self.optimizer.minimize(self.loss, global_step=self.global_step)

    def predict(self, sess, d):
        return sess.run(self.predictions, {self.X: d})

    def update(self, sess, x, a, r):
        _, loss = sess.run([self.train_op, self.loss], feed_dict={self.X:x, self.actions:a, self.rewards:r})
        return loss

def create_initial_state(sess, processor):
    state = env.reset()

    # 1. PROCESS STATE(IMG)

    # 2. STACK 4 STATES TO CREATE STATE

    return state

def update_next_state(state, next_state, sess, processor):
    # 1. PROCESS NEXT STATE(IMG)

    # 2. ADD NEXT_STATE TO STACK, REMOVE FIRST STATE FROM STACK

    return next_state


def collect_experience(sess, replay_memory, processor, epsilons, policy, global_t, epsilon_decay_steps):
    buffer = deque()
    state = create_initial_state(sess, processor)

    for i in range(replay_memory):

        #List of len 4, sum to 1.
        #action_probs = policy(sess, state, epsilons[min(global_t, epsilon_decay_steps - 1)])

        #Will choose from the probs based on prob distribution
        action = 0#np.random.choice(len(action_probs), p=action_probs)

        next_state, reward, done, _ = env.step(VALID_ACTIONS[action])
        next_state = update_next_state(state, next_state, sess, processor)

        buffer.append((state, action, reward, next_state, done))

        state = next_state

        if done:
            state = create_initial_state(sess, processor)

    return buffer


def copy_model(sess, q_model, t_model):
    q_params = [i for i in tf.trainable_variables() if i.name.startswith(q_model.scope)]
    q_params = sorted(q_params, key=lambda v:v.name)

    t_params = [i for i in tf.trainable_variables() if i.name.startswith(t_model.scope)]
    t_params = sorted(t_params, key=lambda v:v.name)

    updated_ops = []

    for q, t in zip(q_params, t_params):
        op = t.assign(q)
        updated_ops.append(op)

    sess.run(updated_ops)

def run():
    batch = 32
    discount = 0.99
    epsilon_start = 1.0
    epsilon_end = 0.05
    epsilon_decay_steps = 500000
    buffer_size = 50000
    buffer_max_size = 100000
    update_target = 5000
    train_episodes = 10000

    tf.reset_default_graph()

    experiment_dir = os.path.abspath("./data/{}".format(env.spec.id))
    checkpoint_dir = os.path.join(experiment_dir, 'checkpoints')
    checkpoint_path = os.path.join(checkpoint_dir, "model")

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    q_model = Model(scope="q", batch=batch)

    target_model = Model(scope='target_q', batch=batch)
    processor = StateProcessor()
    policy = epsilon_greedy(q_model, len(VALID_ACTIONS))

    #Create a distribution of all the steps from start to end [1.0 ... 0.01] based on decay steps
    epsilons = np.linspace(epsilon_start, epsilon_end, epsilon_decay_steps)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()

        chkpoint = tf.train.latest_checkpoint(checkpoint_dir)
        if chkpoint:
            print("loading saved model  ")
            saver.restore(sess, chkpoint)

        global_step = q_model.global_step.eval()

        buffer = collect_experience(sess, buffer_size, processor, epsilons, policy, global_step, epsilon_decay_steps)

        start_time = time.time()
        max_score = 0

        for i in range(train_episodes):
            saver.save(tf.get_default_session(), checkpoint_path)

            state = create_initial_state(sess, processor)

            game_score = 0

            while "Thor" > "Thanos":
                epsilon = epsilons[min(global_step, epsilon_decay_steps - 1)]

                if global_step % update_target == 0:
                    copy_model(sess, q_model, target_model)

                #actions_prob = policy(sess, state, epsilon)

                action = 0#np.random.choice(len(actions_prob), p=actions_prob)

                next_state, reward, done, _ = env.step(VALID_ACTIONS[action])
                next_state = update_next_state(state, next_state, sess, processor)
                game_score += reward

                if len(buffer) > buffer_max_size:
                    buffer.popleft()

                buffer.append((state, action, reward, next_state, done))

                # 1. CREATE MINI BATCH OF SIZE 32 FROM BUFFER


                # 2. GET Q_VALS FROM NEXT STATES (S', A')


                # 3. CALCULATE LABELS FOR TRAINING


                # 4. LEARN, GET LOSS


                #if global_step % 200 == 0:
                    #print("ITER:  ", global_step, " EPSILON  ", epsilon, " LOSS  ", loss, "  REWARDS  ", sum(target))

                if done:
                    break

                global_step += 1
                state = next_state

            if game_score > max_score:
                max_score = game_score
            game_time = float(time.time()-start_time)
            print("GAME Score ", game_score, "MAX SCORE ", max_score, " TIME  ", game_time/60)


def play_game():
    #env = gym.make('Breakout-v0')

    experiment_dir = os.path.abspath("./data/{}".format(env.spec.id))
    checkpoint_dir = os.path.join(experiment_dir, 'checkpoints')

    q_model = Model(scope="q", batch=1)
    processor = StateProcessor()
    policy = epsilon_greedy(q_model, len(VALID_ACTIONS))

    # Create a distribution of all the steps from start to end [1.0 ... 0.01] based on decay steps

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()

        chkpoint = tf.train.latest_checkpoint(checkpoint_dir)
        if chkpoint:
            print("loading saved model  ")
            saver.restore(sess, chkpoint)
        state = create_initial_state(sess, processor)

        while True:
            env.render()
            #action_probs = policy(sess, state, 0)
            action = env.action_space.sample()#np.random.choice(len(VALID_ACTIONS), p=action_probs)

            next_state, reward, done, _ = env.step(action)
            next_state = update_next_state(state, next_state, sess, processor)
            state = next_state
            if done:
                state = create_initial_state(sess, processor)

            #time.sleep(0.0075)

    return

if __name__=="__main__":

    run()
    #play_game()