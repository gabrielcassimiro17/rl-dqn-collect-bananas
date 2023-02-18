# rl-dqn-collect-bananas
Repository for the Udacity RL Specialization first project with Deep Q Learning

# Project overview
This project has the objective to train an Agent using Deep Q Networks.

## Enviroment & Task
The environment consists in a square world with yellow and blue bananas. The agent has the objective to collect as many yellow bananas as possible while avoiding the blue ones. The agent has 4 possible actions: move forward, move backward, turn left and turn right.
The state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around the agent's forward direction. A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana.
The task is episodic, and in order to solve the environment, the agent must get an average score of +13 over 100 consecutive episodes.

<div align="center">
    <img src="https://video.udacity-data.com/topher/2018/June/5b1ab4b0_banana/banana.gif">
</div>

# Usage

## Installing the environment

To install the env, select the environment that matches your operating system:

- [Linux](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)
- [Mac OSX](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)
- [Windows (32-bit)](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip)
- [Windows (64-bit)](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)

The Github Repo already contains the environment for MacOS. If you are using another OS, you must download the environment and place it in the folder `rl-dqn-collect-bananas/`.

## Training
To train the agent you must open the notebook `Navigation.ipynb` and run all the cells. The agent will be trained and the weights will be saved in the file `model.pth`.


## Visualizing trained agent
To visualize the trained agent you must open the notebook `Play.ipynb` and run all the cells.

# Dependencies
The dependencies are listed in the file `requirements.txt` in the folder `python/`. To install them you can run the following command:

```bash
cd python
pip install .
```
ps: This command is in the first cell of the notebook. You should run it just once.

It is highly recommended to use a virtual environment to install the dependencies. you can do this by running the following commands:

	- Linux or Mac:
	```bash
	conda create --name drlnd python=3.6
	source activate drlnd
	```
	- Windows:
	```bash
	conda create --name drlnd python=3.6
	activate drlnd
	```
