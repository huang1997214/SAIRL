import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from collections import deque
import tensorflow.compat.v1 as tf

'''
Tensorflow Setting
'''
tf.disable_eager_execution()
tf.disable_v2_behavior()
random.seed(6)
np.random.seed(6)
tf.set_random_seed(6)

def js_div(p_output, q_output, get_softmax=False):
    """
    Function that measures JS divergence between target and output logits:
    """
    KLDivLoss = nn.KLDivLoss(reduction='batchmean')
    if get_softmax:
        p_output = F.softmax(p_output)
        q_output = F.softmax(q_output)
    log_mean_output = ((p_output + q_output) / 2).log()
    return (KLDivLoss(log_mean_output, p_output) + KLDivLoss(log_mean_output, q_output)) / 2

class RL_model(nn.Module):
    def __init__(self, in_channels, num_actions, args, memory_size=800, batch_size=32):
        super(RL_model, self).__init__()
        self.num_actions = num_actions
        self.batch_size = batch_size
        self.in_channels = in_channels
        self.memory_size = memory_size
        self.replay_buffer = deque()
        self.Dqn = Dueling_DQN(in_channels, num_actions)
        self.target_Dqn = Dueling_DQN(in_channels, num_actions)
        self.gamma=0.99
        self.optimizer = torch.optim.Adam(self.Dqn.parameters(), lr=args.Lr_DDQN)
        self.criterion = nn.L1Loss()
        self.dis = Dis(in_channels, num_actions)
        self.Dis_buffer = deque()
        self.dis_optimizer = torch.optim.Adam(self.dis.parameters(), lr=args.Lr_Dis)
        self.dis_criterion = nn.CrossEntropyLoss()
        self.sigma = args.sigma

    def forward(self, x):
        x = self.Dqn(x)
        return x

    def update_target(self):
        self.target_Dqn.load_state_dict(self.Dqn.state_dict())

    def learn(self, episode):
        minibatch = random.sample(self.replay_buffer, self.batch_size)
        state_batch = torch.from_numpy(np.array([data[0] for data in minibatch])).type(torch.FloatTensor)
        action_batch = torch.from_numpy(np.array([data[1] for data in minibatch])).type(torch.LongTensor)
        reward_batch = torch.from_numpy(np.array([data[2] for data in minibatch])).type(torch.FloatTensor)
        next_state_batch = torch.from_numpy(np.array([data[3] for data in minibatch])).type(torch.FloatTensor)
        q_values = self.Dqn(state_batch)
        aa, _ = q_values.max(1)
        q_s_a = q_values.gather(1, action_batch.unsqueeze(1))
        q_s_a = q_s_a.squeeze()
        q_tp1_values = self.target_Dqn(next_state_batch).detach()
        q_s_a_prime, a_prime = q_tp1_values.max(1)
        expert_act = self.dis(state_batch)
        dis_loss = js_div(q_values, expert_act)
        error = self.criterion(q_s_a, reward_batch + self.gamma*q_s_a_prime)
        dis_loss = self.sigma * dis_loss
        if episode>=3:
            error += dis_loss
        elif episode>5:
            error += dis_loss/2
        self.optimizer.zero_grad()
        error.backward()
        self.optimizer.step()

    def ad_learn_dis(self):
        minibatch = random.sample(self.replay_buffer, self.batch_size)
        state_batch = torch.from_numpy(np.array([data[0] for data in minibatch])).type(torch.FloatTensor)
        q_values = self.Dqn(state_batch)
        aa, _ = q_values.max(1)
        expert_act = self.dis(state_batch)
        dis_loss = js_div(q_values, expert_act)
        dis_loss = -1 * self.sigma * dis_loss
        self.dis_optimizer.zero_grad()
        dis_loss.backward()
        self.dis_optimizer.step()

    def store_buffer(self, state, action, reward, _state):
        one_hot_action = np.zeros(self.num_actions)
        one_hot_action[action] = 1
        self.replay_buffer.append((state, action, reward, _state))
        if len(self.replay_buffer) > self.memory_size:
            self.replay_buffer.popleft()

    def store_dis_buffer(self, state, action, reward, _state):
        one_hot_action = np.zeros(self.num_actions)
        one_hot_action[action] = 1
        self.Dis_buffer.append((state, action, reward, _state))
        if len(self.Dis_buffer) > self.memory_size:
            self.Dis_buffer.popleft()

    def dis_learn(self):
        minibatch = random.sample(self.Dis_buffer, self.batch_size)
        state_batch = torch.from_numpy(np.array([data[0] for data in minibatch])).type(torch.FloatTensor)
        action_batch = torch.from_numpy(np.array([data[1] for data in minibatch])).type(torch.LongTensor)
        q_values = self.Dqn(state_batch)
        loss = self.dis_criterion(q_values, action_batch)
        self.dis_optimizer.zero_grad()
        loss.backward()
        self.dis_optimizer.step()

class myDQN(nn.Module):
    def __init__(self, in_channels, num_actions):
        super(myDQN, self).__init__()
        self.fc1 = nn.Linear(in_features=in_channels, out_features=20)
        #self.fc2 = nn.Linear(in_features=18, out_features=18)
        self.fc3 = nn.Linear(in_features=20, out_features=num_actions)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        #x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class Dis(nn.Module):
    def __init__(self, in_channels, num_actions):
        super(Dis, self).__init__()
        self.fc1 = nn.Linear(in_features=in_channels, out_features=20)
        self.fc2 = nn.Linear(in_features=20, out_features=20)
        self.fc3 = nn.Linear(in_features=20, out_features=num_actions)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        #x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        return x

class Dueling_DQN(nn.Module):
    def __init__(self, in_channels, num_actions):
        super(Dueling_DQN, self).__init__()
        self.num_actions = num_actions
        self.fc1_adv = nn.Linear(in_features=in_channels, out_features=20)
        self.fc2_adv = nn.Linear(in_features=20, out_features=num_actions)
        self.fc1_val = nn.Linear(in_features=in_channels, out_features=20)
        self.fc2_val = nn.Linear(in_features=20, out_features=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        if len(x.shape) == 1:
            x = x.unsqueeze(0)
        adv = self.relu(self.fc1_adv(x))
        val = self.relu(self.fc1_val(x))
        adv = self.fc2_adv(adv)
        val = self.fc2_val(val).expand(x.size(0), self.num_actions)
        x = val + adv - adv.mean(1).unsqueeze(1).expand(x.size(0), self.num_actions)
        x = x.squeeze()
        return x

class baseline_DQN:
    def __init__(
            self,
            n_actions,
            n_features,
            learning_rate=0.01,
            reward_decay=0.9,
            e_greedy=0.9,
            replace_target_iter=50,
            memory_size=800,
            batch_size=30,
            e_greedy_increment=0.002,
            # output_graph=False,
    ):
        self.n_actions = n_actions   # if +1: allow to reject jobs
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon_max = e_greedy
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.epsilon_increment = e_greedy_increment
        self.epsilon = 0.01 if e_greedy_increment is not None else self.epsilon_max

        self.learn_step_counter = 0  # total learning step
        self.replay_buffer = deque()  # init experience replay [s, a, r, s_, done]

        # consist of [target_net, evaluate_net]
        self._build_net()
        t_params = tf.get_collection('target_net_params')
        e_params = tf.get_collection('eval_net_params')

        self.replace_target_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def _build_net(self):
        w_initializer = tf.random_normal_initializer(0., 0.3, 5)  # (mean=0.0, stddev=1.0, seed=None)
        # w_initializer = tf.random_normal_initializer(0., 0.3)  # no seed
        b_initializer = tf.constant_initializer(0.1)
        n_l1 = 20  # config of layers

        # ------------------ build evaluate_net ------------------
        self.s = tf.placeholder(tf.float32, [None, self.n_features], name='s')  # input
        with tf.variable_scope('eval_net', reuse=tf.AUTO_REUSE):
            # c_names(collections_names) are the collections to store variables
            c_names = ['eval_net_params', tf.GraphKeys.GLOBAL_VARIABLES]

            # first layer
            with tf.variable_scope('l1'):
                w1 = tf.get_variable('w1', [self.n_features, n_l1], initializer=w_initializer, collections=c_names)
                b1 = tf.get_variable('b1', [1, n_l1], initializer=b_initializer, collections=c_names)
                l1 = tf.nn.relu(tf.matmul(self.s, w1) + b1)
            # second layer
            with tf.variable_scope('l2'):
                w2 = tf.get_variable('w2', [n_l1, self.n_actions], initializer=w_initializer, collections=c_names)
                b2 = tf.get_variable('b2', [1, self.n_actions], initializer=b_initializer, collections=c_names)
                self.q_eval = tf.matmul(l1, w2) + b2

        # --------------------calculate loss---------------------
        self.action_input = tf.placeholder("float", [None, self.n_actions])
        self.q_target = tf.placeholder(tf.float32, [None], name='Q_target')  # for calculating loss
        q_evaluate = tf.reduce_sum(tf.multiply(self.q_eval, self.action_input), reduction_indices=1)
        with tf.variable_scope('train', reuse=tf.AUTO_REUSE):
            self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, q_evaluate))
            self._train_op = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)
            print('xxxasdasdasd',self.loss)
            # self._train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss)  # better than RMSProp

            # ------------------ build target_net ------------------
        self.s_ = tf.placeholder(tf.float32, [None, self.n_features], name='s_')  # input
        with tf.variable_scope('target_net', reuse=tf.AUTO_REUSE):
            # c_names(collections_names) are the collections to store variables
            c_names = ['target_net_params', tf.GraphKeys.GLOBAL_VARIABLES]

            # first layer
            with tf.variable_scope('l1'):
                w1 = tf.get_variable('w1', [self.n_features, n_l1], initializer=w_initializer, collections=c_names)
                b1 = tf.get_variable('b1', [1, n_l1], initializer=b_initializer, collections=c_names)
                l1 = tf.nn.relu(tf.matmul(self.s_, w1) + b1)
            # second layer
            with tf.variable_scope('l2'):
                w2 = tf.get_variable('w2', [n_l1, self.n_actions], initializer=w_initializer, collections=c_names)
                b2 = tf.get_variable('b2', [1, self.n_actions], initializer=b_initializer, collections=c_names)
                self.q_next = tf.matmul(l1, w2) + b2

            # print('w1:', w1, '  b1:', b1, ' w2:', w2, ' b2:', b2)

    def choose_action(self, state):
        pro = np.random.uniform()
        if pro < self.epsilon:
            actions_value = self.sess.run(self.q_eval, feed_dict={self.s: [state]})
            action = np.argmax(actions_value)
            # print('pro: ', pro, ' q-values:', actions_value, '  best_action:', action)
            # print('  best_action:', action)
        else:
            action = np.random.randint(0, self.n_actions)
            # print('pro: ', pro, '  rand_action:', action)
            # print('  rand_action:', action)
        return action

    def choose_best_action(self, state):
        actions_value = self.sess.run(self.q_eval, feed_dict={self.s: [state]})
        action = np.argmax(actions_value)
        return action

    def store_transition(self, s, a, r, s_):
        one_hot_action = np.zeros(self.n_actions)
        one_hot_action[a] = 1
        self.replay_buffer.append((s, one_hot_action, r, s_))
        if len(self.replay_buffer) > self.memory_size:
            self.replay_buffer.popleft()

    def learn(self):
        # check to replace target parameters
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.sess.run(self.replace_target_op)
            # print('-------------target_params_replaced------------------')

        # sample batch memory from all memory: [s, a, r, s_]
        minibatch = random.sample(self.replay_buffer, self.batch_size)
        state_batch = [data[0] for data in minibatch]
        action_batch = [data[1] for data in minibatch]
        next_state_batch = [data[3] for data in minibatch]

        # calculate target q-value batch
        q_next_batch = self.sess.run(self.q_next, feed_dict={self.s_: next_state_batch})
        q_real_batch = []
        for i in range(self.batch_size):
            q_real_batch.append(minibatch[i][2] + self.gamma * np.max(q_next_batch[i]))
        # train eval network
        self.sess.run(self._train_op, feed_dict={
            self.s: state_batch,
            self.action_input: action_batch,
            self.q_target: q_real_batch
        })

        # increasing epsilon
        if self.epsilon < self.epsilon_max:
            self.epsilon = self.epsilon + self.epsilon_increment
        else:
            self.epsilon = self.epsilon_max
        # print('epsilon:', self.epsilon)
        self.learn_step_counter += 1

class baselines:
    def __init__(self, n_actions, VMtypes):
        self.n_actions = n_actions
        self.VMtypes = np.array(VMtypes)  # change list to numpy
        # parameters for sensible policy
        self.sensible_updateT = 5
        self.sensible_counterT = 1
        self.sensible_discount = 0.7  # 0.7 is best, 0.5 and 0.6 OK
        self.sensible_W = np.zeros(self.n_actions)
        self.sensible_probs = np.ones(self.n_actions) / self.n_actions
        self.sensible_probsCumsum = self.sensible_probs.cumsum()
        self.sensible_sumDurations = np.zeros((2, self.n_actions))  # row 1: jobNum   row 2: sum duration

    def random_choose_action(self):  # random policy
        action = np.random.randint(self.n_actions)  # [0, n_actions)
        return action

    def RR_choose_action(self, job_count):  # round robin policy
        action = (job_count-1) % self.n_actions
        return action

    def early_choose_action(self, idleTs):  # earliest policy
        action = np.argmin(idleTs)
        return action

    def suit_choose_action(self, attrs):  # suitable policy--best
        jobType = attrs[0]  # e.g. 1
        idleTimes = attrs[1:len(attrs)]
        judge = np.argwhere(self.VMtypes == jobType)  # e.g. [[5],[6],[7],[8],[9]]
        judgeF = judge.reshape((len(judge)))  # e.g. [5,6,7,8,9]
        idleTimes_suit = [idleTimes[w] for w in judgeF]  # e.g. [0.2, 0.1, 0.3, 0.5, 1]
        id = idleTimes_suit.index(min(idleTimes_suit))  # e.g. 1
        action = judgeF[id]  # 6
        return action

    def sensible_choose_action(self, arrivalT):  # sensible routing policy
        # if need update prob

        if arrivalT >= self.sensible_updateT * self.sensible_counterT:
            # temp_W = self.sensible_sumDurations[1, :] / self.sensible_sumDurations[0, :]
            temp_W = self.sensible_sumDurations[1, :]
            where_are_inf = np.isinf(temp_W)  # if no job on some VMs, set their duraiton = 0
            where_are_nan = np.isnan(temp_W)
            temp_W[where_are_inf] = 0
            temp_W[where_are_nan] = 0
            # update prob
            self.sensible_W = (1-self.sensible_discount)*self.sensible_W + self.sensible_discount*temp_W
            sensible_W_temp = 1/self.sensible_W
            where_are_inf = np.isinf(sensible_W_temp)  # if no job on some VMs, set their duraiton = 0
            where_are_nan = np.isnan(sensible_W_temp)
            sensible_W_temp[where_are_inf] = 0
            sensible_W_temp[where_are_nan] = 0
            self.sensible_probs = sensible_W_temp/sum(sensible_W_temp)
            self.sensible_probsCumsum = self.sensible_probs.cumsum()

            # initial paras
            self.sensible_counterT += 1
            self.sensible_sumDurations = np.zeros((2, self.n_actions))

        # choose action
        pro = np.random.uniform()
        action = 0
        for i in range(self.n_actions):
            if pro < self.sensible_probsCumsum[i]:
                break
            else:
                action += 1
        return action

    def sensible_counter(self, duration, VMid):
        self.sensible_sumDurations[0, VMid] += 1
        self.sensible_sumDurations[1, VMid] += duration

    def sensible_reset(self):
        self.sensible_W = np.zeros(self.n_actions)
        self.sensible_probs = np.ones(self.n_actions) / self.n_actions
        self.sensible_probsCumsum = self.sensible_probs.cumsum()
        self.sensible_sumDurations = np.zeros((2, self.n_actions))
        self.sensible_counterT = 1