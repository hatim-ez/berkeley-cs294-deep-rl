import matplotlib.pyplot as plt
from run_behavioral_cloning import *
import os

def generate_rollouts(model, envname, num_rollouts):
    env = gym.make(envname)
    max_steps = env.spec.timestep_limit

    returns = []
    for i in range(num_rollouts):
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
            if steps % 100 == 0:
                print("%i/%i" % (steps, max_steps))
            if steps >= max_steps:
                break
        returns.append(totalr)

    print('returns', returns)
    print('mean return', np.mean(returns))
    print('std of return', np.std(returns))

    return returns

def main():
    epochs = [1, 3, 5, 10, 15, 20, 25, 30]
    mean_return = []
    stds = []

    BATCH_SIZE = 32
    dirname = os.path.dirname(__file__)
    for epoch in epochs:
        observations, actions = load_data('HalfCheetah-v2')

        model = create_train_model(observations, actions, epoch, BATCH_SIZE)

        model.save(os.path.join(dirname,
                                'model',
                                'model_HalfCheetah-v2_keras.h5'
                                )
                   )

        model = keras.models.load_model(os.path.join(dirname,
                                                     'model',
                                                     'model_HalfCheetah-v2_keras.h5'
                                                     )
                                        )

        returns = generate_rollouts(model, 'HalfCheetah-v2', 20)
        mean_return.append(np.mean(returns))
        stds.append(np.std(returns))


    plt.plot(epochs, mean_return)
    plt.suptitle('Behavorial Cloning - Mean reward = f(number of epoch)')
    plt.xlabel('Number of epochs in training')
    plt.xlim([0,31])
    plt.ylabel('Mean reward')
    plt.errorbar(epochs, mean_return, yerr=stds)
    plt.savefig('figures/Q2.3.png')

if __name__ == '__main__':
    main()
