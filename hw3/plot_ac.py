
from pylab import *
import os
from os.path import join

dirnames = os.listdir("data")
fig, ax = plt.subplots(figsize=(25,20))

ax.set_title("Finding best target update per gradient steps ratio for actor-critic on cartpole")
ax.set_xlabel("Iteration")
ax.set_ylabel("Average return")

for dirname in dirnames:
    if not ("CartPole" in dirname) or dirname == "ac_1_1_CartPole-v0_10-10-2018_20-40-20":
        continue
    print(dirname)
    experiments = {}
    for subdirname in os.listdir(os.path.join("data", dirname)):
        experiments[subdirname] = np.genfromtxt(join("data", dirname, subdirname, 'log.txt'), delimiter='\t', dtype=None, names=True)
    for key, data in experiments.items():
        errorbar(data['Iteration'], data['AverageReturn'], np.sqrt(data['StdReturn']), label=dirname[3:8] + "_{}".format(key))
ax.legend(prop={'size': 10}).draggable()
plt.savefig("figs/ac_update_settings_comparison.png")
#plt.show()



fig, ax = plt.subplots(figsize=(25,20))

ax.set_title("AC results on Inverted Pendulum for ntu=10, ngsptu=10")
ax.set_xlabel("Iteration")
ax.set_ylabel("Average return")

for dirname in dirnames:
    if not ("InvertedPendulum" in dirname):
        continue
    print(dirname)
    experiments = {}
    for subdirname in os.listdir(os.path.join("data", dirname)):
        experiments[subdirname] = np.genfromtxt(join("data", dirname, subdirname, 'log.txt'), delimiter='\t', dtype=None, names=True)
    for key, data in experiments.items():
        errorbar(data['Iteration'], data['AverageReturn'], np.sqrt(data['StdReturn']), label=dirname[3:8] + "_{}".format(key))
ax.legend(prop={'size': 10}).draggable()
plt.savefig("figs/ac_inverted_pendulum_results.png")


fig, ax = plt.subplots(figsize=(25,20))


ax.set_title("AC results on Inverted Pendulum for ntu=10, ngsptu=10")
ax.set_xlabel("Iteration")
ax.set_ylabel("Average return")


for dirname in dirnames:
    if not ("HalfCheetah" in dirname):
        continue
    print(dirname)
    experiments = {}
    for subdirname in os.listdir(os.path.join("data", dirname)):
        experiments[subdirname] = np.genfromtxt(join("data", dirname, subdirname, 'log.txt'), delimiter='\t', dtype=None, names=True)
    for key, data in experiments.items():
        errorbar(data['Iteration'], data['AverageReturn'], np.sqrt(data['StdReturn']), label=dirname[3:8] + "_{}".format(key))
ax.legend(prop={'size': 10}).draggable()
plt.savefig("figs/ac_half_cheetah_results.png")