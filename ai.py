import tensorflow as tf
import numpy as np

tf.logging.set_verbosity(tf.logging.INFO)

session = tf.InteractiveSession()

def run_session():
    session.run(tf.initialize_all_variables())

class AI:

    def __init__(self, ACTIONS):

        #[BATCH, IMG_W, IMG_H, CHANNELS]
        self.observations = tf.placeholder(shape=(None, 5, 5, 1), dtype=tf.float32)

        #[BATCH, ACTION]
        self.actions = tf.placeholder(dtype=tf.float32, shape=(None, ACTIONS))

        # [BATCH]
        self.rewards = tf.placeholder(dtype=tf.int32, shape=(None))

        self.build_model(ACTIONS)



    def train(self, feed_dict):
        loss = session.run([self.train_op], feed_dict=feed_dict)
        return loss

    def predict(self, feed_dict):
        return self.logits.eval(feed_dict=feed_dict) # add keep_prob 1.0 for dropout

    def build_model(self, ACTIONS):

        # last item in the array (64) is the Number of filters
        # [Filter w, filter h, channels, num_filters]
        w_conv1 = tf.Variable(tf.truncated_normal([2, 2, 1, 32]))
        b_conv1 = tf.Variable(tf.constant(0.01, shape=[1]))

        w_fc1 = tf.Variable(tf.truncated_normal([128, ACTIONS]))
        b_fc1 = tf.Variable(tf.constant(0.01, shape=[ACTIONS]))


        conv1 = tf.nn.relu(tf.nn.conv2d(
            self.observations,
            w_conv1,  # FILTER? THE WINDOW
            # [BATCH, WIDTH, HEIGHT, CHANNELS(DEPTH)],
            strides=[1, 1, 1, 1],  # RESPONSIBLE FOR THE JUMPS/SLIDES SPACES MOVED EACH SHIFT
            padding='VALID'
        )+b_conv1)

        # WITH POOLING THE STRIDE SHOULD NOT OVERLAP
        pool1 = tf.nn.max_pool(
            conv1,
            ksize=[1, 2, 2, 1],
            strides=[1, 2, 2, 1],
            padding="VALID"
        )

        pool1_flat = tf.reshape(pool1, [-1, 2 * 2 * 32])

        self.logits = tf.nn.relu(tf.matmul(pool1_flat, w_fc1) + b_fc1)
        #print("CONV SHAPE  ", conv1.shape, " POOL SHAPE  ", pool1.shape, "  dense  ", logits)

        #ADD
            #1. Dropout
            #2. 2nd Dense layer

        self.predictions = {
            "classes":tf.argmax(self.logits),
            "probabilities":tf.nn.softmax(self.logits, name='softmax_tensor')
        }

        action_logits = tf.reduce_mean(tf.matmul(self.logits, tf.transpose(self.actions)), axis=0)


        loss = tf.losses.sparse_softmax_cross_entropy(labels=self.rewards, logits=action_logits)
        self.reduced_loss = tf.reduce_sum(loss)

        optimizer = tf.train.RMSPropOptimizer(learning_rate=0.01, decay=0.9)

        self.train_op = optimizer.minimize(loss=self.reduced_loss, global_step=tf.train.get_global_step())

        #print("ACTION LOGITS ", action_logits.shape, " losss  ", loss.shape)



    def conv2d_B(self):
        conv1 = tf.layers.conv2d(
            inputs=self.observations,
            filters=32,
            kernel_size=[2, 2],
            padding="valid",
            activation=tf.nn.relu)

        # [BATCH, WIDTH, HEIGHT, FILTERS] as shape
        pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

        pool1_flat = tf.reshape(pool1, [-1, 2 * 2 * 32])

        #Can set units to number of classes and use as logits
        dense = tf.layers.dense(inputs=pool1_flat, units=512, activation=tf.nn.relu)

        #dropout = tf.layers.dropout(inputs=dense, rate=0.4)
        #logits = tf.layers.dense(inputs=dropout, units=10)

        #prediction_raw =  tf.argmax(dense) #Corrosponding classes not as probability

        #Softmax creates class probability
        prediction_prob = tf.nn.softmax(dense, name='softmax_tensor')


        #Loss

        print("CONV LAYER SHPAE ", conv1.shape, "MAX POOL SIZE  ", pool1.shape, " pool flat   ", pool1_flat, " dense ", dense )



if __name__ == "__main__":

    #mnist = tf.contrib.learn.datasets.load_dataset("mnist")
    #train_data = mnist.train.images  # Returns np.array
    #train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
    #eval_data = mnist.test.images  # Returns np.array
    #eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)
    #estimator = tf.estimator.Estimator(model_fn=model, model_dir="/tmp/mnist_convnet_model")
    #tensors_to_log = {"probabilities": "softmax_tensor"}

    ai = AI(4)







