#!/usr/bin/env python

"""
DAgger implemented to clone an expert policy.
Example usage:
    python run_dagger.py experts/HalfCheetah-v2.pkl HalfCheetah-v2 \
        --dagger_iter 20 --render --num_rollouts 20
"""

import pickle
import numpy as np
import tensorflow as tf
import tf_util
import gym
import load_policy
from keras.models import Sequential, load_model
from keras.layers import Dense
import argparse
import os
from sklearn.model_selection import train_test_split


def load_data(filename):
    with open(filename, 'rb') as f:
        data = pickle.loads(f.read())
    return data


def main():
    dirname = os.path.dirname(__file__)

    parser = argparse.ArgumentParser()
    parser.add_argument('expert_policy_file', type=str)
    parser.add_argument('envname', type=str)
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--num_rollouts', type=int, default=20,
                        help='Number of expert roll outs')
    parser.add_argument('--dagger_iter', type=int, default=5,
                        help='Number of dagger iterations')
    parser.add_argument("--max_timesteps", type=int)
    args = parser.parse_args()

    print('loading and building expert policy')
    policy_fn = load_policy.load_policy(args.expert_policy_file)
    print('loaded and built')

    envname = args.envname
    mean_rewards = []
    stds = []

    print('loading data from expert policy')
    with open(os.path.join("expert_data", "{0}" + ".pkl").format(envname), 'rb') as f:
        data = pickle.loads(f.read())
    observations = np.array(data['observations'])
    actions = np.array(data['actions'])
    actions = np.squeeze(actions)

    model = Sequential([
        Dense(100, input_shape=(observations.shape[-1],), activation='relu'),
        Dense(100, activation='relu'),
        Dense(actions.shape[-1])
    ])
    model.compile(loss='msle', optimizer='adam', metrics=['accuracy'])
    model.save(os.path.join(dirname, 'model', 'model_{0}_dagger_keras.h5'.format(envname)))

    # Main DAGGER Loop
    for i in range(args.num_rollouts):
        if i == 0:
            # If i =0, that corresponds to behavioral cloning, and the model has already been trained above.
            continue

        else:
            # 1) Train policy on D
            # Split data into train and test set
            n = observations.shape[0]
            X_train, X_test, y_train, y_test = train_test_split(observations,
                                                                actions,
                                                                train_size=int(n*0.8))

            # Train model on dataset
            model = load_model(os.path.join(dirname, 'model', 'model_{0}_dagger_keras.h5'.format(envname)))
            model.fit(X_train, y_train, batch_size=32, nb_epoch=30, verbose=1)

            model.evaluate(X_test, y_test, verbose=1)
            model.save(os.path.join(dirname, 'model', 'model_{0}_dagger_keras.h5'.format(envname)))

        # 2) Run policy on simulation and 3) Expert labels on these observations
        with tf.Session():
            tf_util.initialize()
            env = gym.make(args.envname)
            max_steps = args.max_timesteps or env.spec.timestep_limit

            returns = []
            new_observations = []
            new_expected_actions = []

            model = load_model(os.path.join(dirname, 'model/model_{0}_dagger_keras.h5'.format(envname)))
            for i in range(args.num_rollouts):
                print('iter', i)
                obs = env.reset()
                done = False
                totalr = 0.
                steps = 0
                while not done:
                    expected_action = policy_fn(obs[None, :])
                    # obs = np.array(obs)
                    # obs = obs.reshape(1, len(obs), 1)
                    action = (model.predict(obs.reshape(1, len(obs) ))) #, batch_size=64, verbose=0))

                    new_observations.append(obs)
                    new_expected_actions.append(expected_action)
                    obs, r, done, _ = env.step(action)
                    totalr += r
                    steps += 1
                    if args.render:
                        env.render()
                    if steps % 100 == 0: print("%i/%i" % (steps, max_steps))
                    if steps >= max_steps:
                        break
                returns.append(totalr)

            print('returns', returns)
            print('mean return', np.mean(returns))
            print('std of return', np.std(returns))
            mean_rewards.append(np.mean(returns))
            stds.append(np.std(returns))

            new_observations = np.array(new_observations)
            new_expected_actions = np.squeeze(np.array(new_expected_actions))

        new_observations = new_observations.reshape((new_observations.shape[0], observations.shape[1]))

        observations = np.concatenate((observations, new_observations))
        actions = np.concatenate((actions, new_expected_actions))

    print(mean_rewards)
    print(stds)
    with open("figures/DAgger_output.txt", "w") as text_file:
        text_file.write("Mean reward per iteration: {0} \n Standard deviations per iteration: {1}".format(mean_rewards, stds))


if __name__ == '__main__':
    main()