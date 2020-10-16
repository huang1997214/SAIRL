import torch
import numpy as np
from env import SchedulingEnv
from model import RL_model, baseline_DQN, baselines
from utils import get_args


args = get_args()
#store result
performance_lamda = np.zeros(args.Baseline_num)
performance_success = np.zeros(args.Baseline_num)
performance_util = np.zeros(args.Baseline_num)
performance_finishT = np.zeros(args.Baseline_num)
#gen env
env = SchedulingEnv(args)
#build model
brainRL = baseline_DQN(env.actionNum, env.s_features)
mod = RL_model(env.s_features, env.actionNum, args)
brainOthers = baselines(env.actionNum, env.VMtypes)

global_step = 0
my_learn_step = 0
for episode in range(args.Epoch):
    print('----------------------------Episode', episode, '----------------------------')
    args.SAIRL_greedy += 0.04
    job_c = 1  # job counter
    performance_c = 0
    env.reset(args)  # attention: whether generate new workload, if yes, don't forget to modify reset() function
    brainOthers.sensible_reset()
    performance_respTs = []
    while True:
        #baseline DQN
        global_step += 1
        finish, job_attrs = env.workload(job_c)
        DQN_state = env.getState(job_attrs, 4)
        if global_step != 1:
                brainRL.store_transition(last_state, last_action, last_reward, DQN_state)
        action_DQN = brainRL.choose_action(DQN_state)  # choose action
        reward_DQN = env.feedback(job_attrs, action_DQN, 4)
        if (global_step > args.Dqn_start_learn) and (global_step % args.Dqn_learn_interval == 0):  # learn
            brainRL.learn()
        last_state = DQN_state
        last_action = action_DQN
        last_reward = reward_DQN

        # my AI DDQN
        my_DQN_state = env.getState(job_attrs, 7)
        if global_step > args.SAIRL_start_learn or episode>=1:
            p = np.random.randint(10)
            if p < args.SAIRL_greedy or episode>=1:
                if episode<2:
                    dev = np.random.rand()
                else:
                    dev = 0
                my_act = mod(torch.FloatTensor(my_DQN_state))

                if dev>0.90:
                    my_act = my_act.detach().numpy()
                    my_act = my_act.argsort()
                    my_act = my_act[-4]
                else:

                    my_act = np.argmax(my_act.detach().numpy())
            else:
                my_act = np.random.randint(10)
        else:
            my_act = np.random.randint(10)
        my_reward_DQN = env.feedback(job_attrs, my_act, 7)
        if global_step != 1:
            mod.store_buffer(my_last_state, my_last_action, my_last_reward, my_DQN_state)
        if (global_step > args.SAIRL_start_learn) and (global_step % args.SAIRL_learn_interval) == 0:
            mod.learn(episode)
            if episode>=3:
                mod.ad_learn_dis()
            my_learn_step += 1
        if my_learn_step % args.SAIRL_update_freq == 0:
            mod.update_target()
        my_last_state = my_DQN_state
        my_last_action = my_act
        my_last_reward = my_reward_DQN
        # random policy
        state_Ran = env.getState(job_attrs, 1)
        action_random = brainOthers.random_choose_action()
        reward_random = env.feedback(job_attrs, action_random, 1)
        # round robin policy
        state_RR = env.getState(job_attrs, 2)
        action_RR = brainOthers.RR_choose_action(job_c)
        reward_RR = env.feedback(job_attrs, action_RR, 2)
        # earliest policy
        idleTimes = env.get_VM_idleT(3)  # get VM state
        action_early = brainOthers.early_choose_action(idleTimes)
        reward_early = env.feedback(job_attrs, action_early, 3)
        # suitable policy
        suit_state = env.getState(job_attrs, 5)  # job type, VM wait time
        action_suit = brainOthers.suit_choose_action(suit_state)  # best
        reward_suit = env.feedback(job_attrs, action_suit, 5)
        # sensible routing policy
        action_sensible = brainOthers.sensible_choose_action(job_attrs[1])  # job_attrs[1]: job arrivalT
        reward_sensible = env.feedback(job_attrs, action_sensible, 6)
        state_sensible = env.getStateP(job_attrs[0])
        brainOthers.sensible_counter(state_sensible, action_sensible)
        #dis buffer
        if global_step != 1:
            mod.store_dis_buffer(last_dis_state, last_dis_action, last_dis_reward, suit_state)
        #slef-paced learning
        if episode == 0:
            if global_step < 2000:
                last_dis_state = state_Ran
                last_dis_action = action_random
                last_dis_reward = reward_random
            else:
                last_dis_state = state_RR
                last_dis_action = action_RR
                last_dis_reward = reward_RR
        else:
            last_dis_state = suit_state
            last_dis_action = action_suit
            last_dis_reward = reward_suit
        #training dis
        if global_step > 100:
            mod.dis_learn()
        if job_c % 500 == 0:
            acc_Rewards = env.get_accumulateRewards(args.Baseline_num, performance_c, job_c)
            finishTs = env.get_FinishTimes(args.Baseline_num, performance_c, job_c)
            avg_exeTs = env.get_executeTs(args.Baseline_num, performance_c, job_c)
            avg_waitTs = env.get_waitTs(args.Baseline_num, performance_c, job_c)
            avg_respTs = env.get_responseTs(args.Baseline_num, performance_c, job_c)
            performance_respTs.append(avg_respTs)
            successTs = env.get_successTimes(args.Baseline_num, performance_c, job_c)
            performance_c = job_c

        job_c += 1
        if episode>2:
            args.SAIRL_learn_interval += 2
        if finish:
            break

    # episode performance
    startP = 2000
    total_Rewards = env.get_totalRewards(args.Baseline_num, startP)
    avg_allRespTs = env.get_total_responseTs(args.Baseline_num, startP)
    total_success = env.get_totalSuccess(args.Baseline_num, startP)
    avg_util = env.get_avgUtilitizationRate(args.Baseline_num, startP)
    total_Ts = env.get_totalTimes(args.Baseline_num, startP)

    print('total performance (after 2000 jobs):')
    for i in range(len(args.Baselines)):
        name = "[" + args.Baselines[i] + "]"
        print(name + " reward:", total_Rewards[i], ' avg_responseT:', avg_allRespTs[i],
              'success_rate:', total_success[i], ' utilizationRate:', avg_util[i], ' finishT:', total_Ts[i])

    if episode != 0:
        performance_lamda[:] += env.get_total_responseTs(args.Baseline_num, 0)
        performance_success[:] += env.get_totalSuccess(args.Baseline_num, 0)
        performance_util[:] += env.get_avgUtilitizationRate(args.Baseline_num, 0)
        performance_finishT[:] += env.get_totalTimes(args.Baseline_num, 0)
print('')


print('---------------------------- Final results ----------------------------')
performance_lamda = np.around(performance_lamda/(args.Epoch-1), 3)
performance_success = np.around(performance_success/(args.Epoch-1), 3)
performance_util = np.around(performance_util/(args.Epoch-1), 3)
performance_finishT = np.around(performance_finishT/(args.Epoch-1), 3)
print('avg_responseT:')
print(performance_lamda)
print('success_rate:')
print(performance_success)
print('utilizationRate:')
print(performance_util)
print('finishT:')
print(performance_finishT)