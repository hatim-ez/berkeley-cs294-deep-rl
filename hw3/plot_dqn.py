from pylab import *
import os
from os.path import join
import pickle

dirname = "dqn_atari"

data = pickle.load(open(os.path.join(dirname, 'Q1_huber_loss_delta=2.pkl'),'rb'))
fig, ax = plt.subplots(figsize=(25,20))

ax.set_title("Graph representing Q-Learning results on Pong for Huber_Loss delta = 2")
ax.set_xlabel("Iteration")
ax.set_ylabel("Return")

plot(data['time_step_log'], data['best_mean_episode_reward_log'], label="best_mean_episode_reward")
plot(data['time_step_log'], data['mean_episode_reward_log'], label="mean_episode_reward")

ax.legend(prop={'size': 10}).draggable()
plt.savefig("figs/Q1_huber_loss_delta=2.png")
#plt.show()


# Compare dqn and vanilla

data_vanilla = pickle.load(open('dqn_atari/Q1_huber_loss_delta=2.pkl', 'rb'))
data_dqn = pickle.load(open('dqn_atari/personal_log_atari_double_q_learning_huber_delta=2.pkl', 'rb'))
fig, ax = plt.subplots(figsize=(25,20))

ax.set_title("Graph comparing Vanilla and Double Q-Learning results on Pong for Huber_Loss delta = 2")
ax.set_xlabel("Iteration")
ax.set_ylabel("Return")

plot(data_vanilla['time_step_log'], data_vanilla['best_mean_episode_reward_log'], label="vanilla_best_mean_episode_reward")
plot(data_vanilla['time_step_log'], data_vanilla['mean_episode_reward_log'], label="vanilla_mean_episode_reward")


plot(data_dqn['time_step_log'], data_dqn['best_mean_episode_reward_log'], label="dqn_best_mean_episode_reward")
plot(data_dqn['time_step_log'], data_dqn['mean_episode_reward_log'], label="dqn_mean_episode_reward")

ax.legend(prop={'size': 10}).draggable()
plt.savefig("figs/Q2_DQN_Vanillahuber_loss_delta=2.png")




dirname = "data_aws"
filenames = os.listdir(dirname)
fig, ax = plt.subplots(figsize=(25,20))

ax.set_title("Graph representing Doube Q-Learning results on Pong for different value of Huber_Loss delta ")
ax.set_xlabel("Iteration")
ax.set_ylabel("Return")

for filename in filenames:
    if not ("personal_log" in filename):
        continue
    print(filename)
    data = pickle.load(open(os.path.join(dirname, filename), 'rb'))
    plot(data['time_step_log'], data['best_mean_episode_reward_log'], label="best_mean_episode_reward" + filename[37:-4])
    plot(data['time_step_log'], data['mean_episode_reward_log'], label="mean_episode_reward" + filename[37:-4])

ax.legend(prop={'size': 10}).draggable()
plt.savefig("figs/dqn_huber_loss_parameter_comparison.png")

