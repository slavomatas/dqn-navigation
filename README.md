# Navigation with DQN Agent

<p align=center>
	<img width=80% src="images/hero.png"/>
</p>

### Introduction


In this project I trained a DQN reinforcement learning agent to reach a score of +13 on 
average over 100 episodes in the Unity-ML Bananas environment. 

In this environment positive reward is accumulated by running into yellow "good" bananas and 
avoiding blue "bad" bananas which return -1 reward. An episode ends after a fixed interval of 300 
steps.

### Report

I have implemented several variants of DQN agents:

1. DQN agent with Experience Reply using vector observations
2. DQN agent with Prioritized Experience Reply using vector observations
3. DQN agent with Prioritized Experience Reply using visual observations
4. Dueling DQN Agent with Prioritized Experience Reply using vector observations
5. Categorical DQN Agent with Experience Replay using vector observations

For details please read the [full report](Report.md).

#### Environment Description

(Text slightly modified from the official course description)

A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided 
for collecting a blue banana.  Thus, the goal of your agent is to collect as many yellow 
bananas as possible while avoiding blue bananas.  

The state space has 37 dimensions and contains the agent's velocity, along with ray-based 
perception of objects around agent's forward direction.  

Ray Perception (35)

7 rays projecting from the agent at the following angles (and returned in this order):

[20, 90, 160, 45, 135, 70, 110] # 90 is directly in front of the agent

Ray (5)

Each ray is projected into the scene. If it encounters one of four detectable objects
the value at that position in the array is set to 1. Finally there is a distance measure
which is a fraction of the ray length.

[Banana, Wall, BadBanana, Agent, Distance]

example

[0, 1, 1, 0, 0.2]

There is a BadBanana detected 20% of the way along the ray and a wall behind it.

Velocity of Agent (2)

- Left/right velocity (usually near 0)
- Forward/backward velocity (0-11.2)

Given this information, the agent has to learn how to best select actions.  Four discrete actions 
are available, corresponding to:
- **`0`** - move forward.
- **`1`** - move backward.
- **`2`** - turn left.
- **`3`** - turn right.

The task is episodic, and in order to solve the environment, your agent must get an average score 
of +13 over 100 consecutive episodes. We discuss the utility of this metric in our report.

### Getting Started

#### Installation

This project has numerous dependencies and assumes you have a working environment according to the
Udacity Deep Reinforcement Learning Nanodegree instructions. If not:

[Install Dependencies Now](https://github.com/udacity/deep-reinforcement-learning#dependencies)

All code for this project is executed from the command line so you can skip Jupyter setup if you'd like.

Next follow the instructions from the [Navigation Project README](https://github.com/udacity/deep-reinforcement-learning/blob/master/p1_navigation/README.md) which we have partially copied below.

1. Download the environment from one of the links below.  You need only select the environment that matches your operating system:
    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)
    - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)
    - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip)
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)
    
    (_For Windows users_) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

    (_For AWS_) If you'd like to train the agent on AWS (and have not [enabled a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md)), then please use [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux_NoVis.zip) to obtain the environment.

2. Place the file in the `unity/` directory of this repository and unzip (or decompress) the file.


### Instructions

The following assume you have a properly installed environment (e.g. a conda env) and are
running these commands from a command line where that environment has been activated.

#### Train

To train an agent from scratch using the provided code run the train script in a respective package.

`python3 train.py`

#### Play

To watch a pre-trained agent perform run the play script in the respective package.

`python play.py`
