# CS294-112 HW 1: Imitation Learning

### Getting started

* rune **python run_expert.py experts/Humanoid-v2.pkl Humanoid-v2 --render --num_rollouts 20** for the different environments you want to work on.

* run **python run_behavioral_cloning.py "HalfCheetah-v2,Humanoid-v2"** to train a behavioral cloning model on expert data (need to run run_expert.py beforehand) for specified environments,
 then run rollouts to generate results and put them in */figures/results_of_behavioral_cloning.html*. 
 Other options include *--max_timesteps*, *--num_rollouts* (set to 20 by default), *render*.
 
 * run **python plot_behavioral_cloning.py** to generate file *figures/Q2.3.png* that plots the evolution of the mean reward according to the number of epochs in training.
 
 * run **python run_dagger.py experts/HalfCheetah-v2.pkl HalfCheetah-v2 \
        --dagger_iter 20 --render --num_rollouts 20** to run DAgger implementation and print results to *figures/DAgger_output.txt*.
        
 * run **python plot_dagger.py** to generate graph representing the evolution of the mean reward and standard deviation according to the number of DAgger iterations for a Humanoid-v2 environment. I chose Humanoid-2, but if you want to try for a different environment, run the previous command and change the *plot_dagger.py* file with the new values from *figures/DAgger_output.txt*

Dependencies:
 * Python **3.5**
 * Numpy version **1.14.5**
 * TensorFlow version **1.10.5**
 * MuJoCo version **1.50** and mujoco-py **1.50.1.56**
 * OpenAI Gym version **0.10.5**

Once Python **3.5** is installed, you can install the remaining dependencies using `pip install -r requirements.txt`.

**Note**: MuJoCo versions until 1.5 do not support NVMe disks therefore won't be compatible with recent Mac machines.
There is a request for OpenAI to support it that can be followed [here](https://github.com/openai/gym/issues/638).

**Note**: Students enrolled in the course will receive an email with their MuJoCo activation key. Please do **not** share this key.

The only file that you need to look at is `run_expert.py`, which is code to load up an expert policy, run a specified number of roll-outs, and save out data.

In `experts/`, the provided expert policies are:
* Ant-v2.pkl
* HalfCheetah-v2.pkl
* Hopper-v2.pkl
* Humanoid-v2.pkl
* Reacher-v2.pkl
* Walker2d-v2.pkl

The name of the pickle file corresponds to the name of the gym environment.

#### Authors

* **Hatim Ezbakhe** - 
* *Jonathan Ho* - Project base: https://github.com/berkeleydeeprlcourse/homework

