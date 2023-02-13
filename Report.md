# Project Report

## Objective
This project has the objective to train an Agent using Deep Q Learning. The agent will be trained to collect yellow bananas while avoiding blue bananas from Unity's Banana Collector environment.
More information about Unix environment can be found [here](https://github.com/Unity-Technologies/ml-agents/blob/main/docs/Learning-Environment-Examples.md#banana-collector).

## Enviroment & Task
The environment consists in a square world with yellow and blue bananas. The agent has the objective to collect as many yellow bananas as possible while avoiding the blue ones. The agent has 4 possible actions: move forward, move backward, turn left and turn right.
The state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around the agent's forward direction. A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana.
The task is episodic, and in order to solve the environment, the agent must get an average score of +13 over 100 consecutive episodes.

<div align="center">
    <img src="https://video.udacity-data.com/topher/2018/June/5b1ab4b0_banana/banana.gif">
</div>

## Implementation
To solve the problem given by the environment it was implemented a Deep Q Learning algorithm. The algorithm is based on the paper [Human-level control through deep reinforcement learning](https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf) by DeepMind.

The algorithm works by using a neural network to approximate the Q-Function. The neural network receives the state as input and outputs the Q-Value for each action. Then it uses the Q-Value to select the best action to be taken by the agent. The algorithm learns by using the Q-Learning algorithm to train the neural network. There are also two problems to a simple implementation of the algorithm: correlated experiences and correlated targets. The algorithm uses two techniques to solve these problems: Experience Replay and Fixed Q-Targets.

### Correlated experiences
Correlated experiences refer to a situation where the experiences (or transitions) of an agent are correlated with each other, meaning they are not independent and identically distributed. This can lead to an overestimation of the expected reward of a particular state or action, resulting in poor performance or convergence to suboptimal policies.

 To solve this problem it is used a technique called Experience Replay. The technique consists in storing the experiences of the agent in a replay buffer and sampling randomly from it to train the neural network.

### Correlated targets
Correlated targets refer to a situation where the target values used to update the policy are not independent of each other, leading to correlation in the learning signal. This can slow down or prevent convergence to the optimal policy.

 To solve this problem it is used a technique called Fixed Q-Targets. The technique consists in using two neural networks: the local network and the target network. The local network is used to select the best action to be taken by the agent while the target network is used to calculate the target value for the Q-Learning algorithm. The target network is updated every 4 steps with the weights of the local network.

## Neural network architecture
The neural network architecture used in the algorithm is a simple fully connected neural network with 2 hidden layers. The input layer has 37 neurons, the output layer has 4 neurons and the hidden layers have **FILLHERE** neurons. The activation function used in the hidden layers is ReLU and the activation function used in the output layer is the identity function.

The optimizer used for this implementation is Adam with a learning rate of **FILLHERE** and a discount factor of **FILLHERE**.

The library used to implement the neural network was PyTorch.





## Future improvements
The algorithm can be improved by using the following techniques:

- Dueling DQN - [paper](https://arxiv.org/pdf/1511.06581.pdf)
- Prioritized Experience Replay - [paper](https://arxiv.org/pdf/1511.05952.pdf)
