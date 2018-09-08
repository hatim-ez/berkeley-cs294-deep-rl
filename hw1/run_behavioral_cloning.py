#!/usr/bin/env python

"""
Code to train a behavioral cloning model on expert data,
 then run rollouts to generate results.
Example usage:
    python run_behavioral_cloning.py "HalfCheetah-v2,Humanoid-v2"
"""

import pickle
from keras.models import Sequential
from keras.layers import Dense
import os
import numpy as np
import argparse
import keras
import gym
import pandas as pd
import codecs



def load_data(envname):
    dirname = os.path.dirname(__file__)
    print('loading data from expert policy')
    with open(os.path.join(dirname, "expert_data", "{0}" + ".pkl").format(envname), 'rb') as f:
        data = pickle.loads(f.read())

    observations = np.array(data["observations"])
    actions = np.array(data["actions"])
    actions = np.squeeze(actions)

    return observations, actions

def create_train_model(input, output, epochs, batch_size):
    model = Sequential([
        Dense(100, input_shape=(input.shape[-1],), activation='relu'),
        Dense(100, activation='relu'),
        Dense(output.shape[-1])
    ])

    model.compile(optimizer='Adam', loss='mse', metrics=['mse'])
    model.fit(input, output,
              validation_split=0.1,
              batch_size=batch_size,
              nb_epoch=epochs,
              verbose=2)

    return model

def generate_rollouts(model, envname, args):
    env = gym.make(envname)
    max_steps = args.max_timesteps or env.spec.timestep_limit

    returns = []
    for i in range(args.num_rollouts):
        print('iter', i)
        obs = env.reset()
        done = False
        totalr = 0.
        steps = 0
        while not done:
            action_pred = model.predict(obs.reshape(1, len(obs)))
            obs, r, done, _ = env.step(action_pred)
            totalr += r
            steps += 1
            if args.render:
                env.render()
            if steps % 100 == 0:
                print("%i/%i" % (steps, max_steps))
            if steps >= max_steps:
                break
        returns.append(totalr)

    print('returns', returns)
    print('mean return', np.mean(returns))
    print('std of return', np.std(returns))

    return returns

def generate_results_table(results):
    columns = ['mean', 'std', 'amount_of_data','n_training_iterations', 'network_size']
    df = pd.DataFrame.from_dict(results, orient='index')
    df.columns = columns
    df[columns[0:4]] = df[columns[0:4]].astype(int)
    df.to_html('figures/results_of_behavioral_cloning.html', columns=columns)
    return


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('envnames', type=str)
    parser.add_argument('--render', action='store_true')
    parser.add_argument("--max_timesteps", type=int)
    parser.add_argument('--num_rollouts', type=int, default=20,
                        help='Number of expert roll outs')
    args = parser.parse_args()

    envnames = args.envnames

    dirname = os.path.dirname(__file__)

    results = {}
    EPOCHS = 30
    BATCH_SIZE = 32
    #max_nb_training_iterations
    for envname in envnames.split(','):
        observations, actions = load_data(envname)

        model = create_train_model(observations, actions, EPOCHS, BATCH_SIZE)

        model.save(os.path.join(dirname,
                                'model',
                                'model_{0}_keras.h5'.format(envname)
            )
        )

        model = keras.models.load_model(os.path.join(dirname,
                                                     'model',
                                                     'model_{0}_keras.h5'.format(envname)
                                                     )
                                        )

        returns = generate_rollouts(model, envname, args)

        number_of_data = observations.shape[0]
        n_training_iterations = number_of_data * EPOCHS / BATCH_SIZE
        results[envname] = [np.mean(returns), np.std(returns), number_of_data, n_training_iterations, "({0}, 100)x(100, 100)x(100, {1})".format(observations.shape[-1], actions.shape[-1])]

    generate_results_table(results)

if __name__ == '__main__':
    main()