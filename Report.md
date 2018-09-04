# Implementing a Banana Obsessed Agent

## Summary

 In this project I trained a DQN reinforcement learning agent to reach a score of +13 on 
 average over 100 episodes in the Unity-ML environment. 
 In this environment positive reward is accumulated by running into yellow "good" bananas and 
 avoiding blue "bad" bananas which return -1 reward. An episode ends after a fixed interval of 300 
 steps.


I have implemented several variants of DQN Agents

# 1. Basic DQN Agent with replay buffer

Basic DQN agent implements Deep Q-Learning Algorithm which has the following steps 

1. Initialize parameters for Q(s, a) and Q ˆ(s, a) with random weights, \epsilon ← 1.0,
and empty replay buffer
2. With probability epsilon, select a random action a, otherwise a = arg max<sub>a</sub> Q <sub>s,a</sub>
3. Execute action in Unity-ML environment and observe reward r and the next state s′
4. Store transition (s, a, r, s′) in the replay buffer
5. Sample a random minibatch of transitions from the replay buffer
6. For every transition in the buffer, calculate target y = r if the episode has
ended at this step or y = r + γ max<sub>a'∈A</sub>a' Q ˆ<sub>s',a'</sub> otherwise
7. Calculate loss: L = (Q<sub>s,a</sub> − y)<sup>2</sup>
8. Update Q(s, a) using the SGD algorithm by minimizing the loss in respect
to model parameters
9. Every 4 steps copy weights from local Q-network to target Q-network ˆ<sub>t</sub>
The target network is frozen for several time steps and then the target network weights are updated 
by copying the weights from the actual Q network. Freezing the target network for a while 
and then updating its weights with the actual Q network weights stabilizes the training.
10. Repeat from step 2 until reaching average reward over last 100 episodes of 13 

The Deep Q-Network architecture of the Basic DQN Agent is following (for vector observations):

```
Fully Connected Layer (128 units)
		  |
		ReLU
		  |
Fully Connected Layer (128 units)
		  |
		ReLU
		  |
Fully Connected Layer (4 units - action size)
```

## 2. DQN Agent with prioritized replay buffer 

The basic DQN used the replay buffer to break the correlation between immediate
transitions in our episodes. The examples we experience during the episode will be highly correlated, as most of the
time the environment is "smooth" and doesn't change much according to our actions.
However, the SGD method assumes that the data we use for training has a i.i.d.
property. To solve this problem, the classic DQN method used a large buffer of
transitions, randomly sampled to get the next training batch.

The main concept of prioritized replay is the criterion by which the importance of each transition is measured. 
One idealised criterion would be the amount the RL agent can learn from a transition in its current state. 
A reasonable proxy for this measure is the magnitude of a transition’s TD error δ, which indicates how ‘surprising’
or unexpected the transition is. 

There are couple of variants of TD-error prioritization. For this project I have used proportional prioritization 
where pi = |δ,<sub>i</sub>| + \epsilon, where \epsilon is a small positive constant that prevents the edge-case of transitions not being revisited once their
error is zero. 

In order to calculate TD-Errors the agent loss calculation has to be implemented explicitly as the MSELoss in PyTorch doesnt support weights. 
Implementing MSE loss explicitly allows us to take into account weights of samples and keep individual loss values for every sample. Those
values will be passed to the priority replay buffer to update priorities. Small values are added to every loss to handle the situation of zero loss value, which will lead
to zero priority of entry.

Snippet of the code for MSE Loss calculation:

losses_v = weights * (Q_expected - Q_targets) ** 2

loss = losses_v.mean()

prios = losses_v + 1e-5
  

The Deep Q-Network architecture of the DQN Agent is following (the same as for basic DQN Agent):

```
Fully Connected Layer (128 units)
		  |
		ReLU
		  |
Fully Connected Layer (128 units)
		  |
		ReLU
		  |
Fully Connected Layer (4 units - action size)
```

## 3. DQN Agent for pixels navigation with prioritized replay buffer
 
## 4. Dueling DQN Agent

The main idea behind Dueling DQN Agent is that the Q-values Q(s, a) Q-network is
trying to approximate can be divided into quantities: the value of the state V(s) and
the advantage of actions in this state A(s, a). The quantity V(s) equals to the 
discounted expected reward achievable from this state. The advantage A(s, a) 
is supposed to bridge the gap from A(s) to Q(s, a), as, by definition: Q(s, a) = V(s) + A(s, a). 
In other words, the advantage A(s, a) is just the delta, saying how much extra reward 
some particular action from the state brings us. Advantage could be positive or negative and, in general, 
can have any magnitude. For example, at some tipping point, the choice of one action over another can cost us
lots of the total reward.

Dueling DQN Agent uses a different Q-Network that processes vector observations (states) using two independent paths: 
one path is responsible for V(s) prediction, which is just a single number, 
and another path predicts individual advantage values, having the same dimension as Q-values in the classic case. 
After that, we add V(s) to every value of A(s, a) to obtain the Q(s, a), which is used and trained as normal.

In order to make sure that the network will learn V(s) and A(s, a) correctly we
have yet another constraint to be set: we want the mean value of the advantage of
any state to be zero.  This constraint could be enforced by subtracting from the Q expression 
in the network the mean value of the advantage, which effectively pulls 
the mean for advantage to zero: <img src="https://latex.codecogs.com/svg.latex?\Large&space;Q(s, a) = V (s) + A(s, a) - \frac{1}{k} \sum A(s, k)" />

This keeps the changes needed to be made in the classic DQN very simple: to convert
it to the double DQN you need to change only the network architecture, without
affecting other pieces of the implementation


## 5. Categorical DQN Agent 



<p align=center>
	<img width=70% src="images/human-2-learning.png"/>
</p>

## Methods

 All training was performed on a single Windows 10 desktop machine with an NVIDIA GTX 970. 

### System Setup

 - Python:			3.6.6
 - CUDA:			9.0
 - Pytorch: 		0.4.1
 - NVIDIA Driver:  	388.13
 - Conda:			4.5.10

### Learning Algorithm

 The agent uses an implementation of the Deep Q-Learning algorithm with experience replay and a 
 target Q Network. 

 It is important to note that both experience replay and the target Q network
 are useful in the context of neural network function approximation because of the 
 independent and identically distributed (IID) assumption built into stochastic gradient update 
 methods. Without these correlation breaking steps networks tend to be biased toward recent
 experience "forgetting" earlier experience which may still be important and representative for
 the task at hand. 

#### Outer Loop

The agent follows the standard State, Action, Reward, State progression for an off-policy 
reinforcement learning algorithm.

 - The agent selects an action in an epsilon-greedy fashion given the current state and an 
   epsilon value
 - That action is passed to the environment which returns
 	- The next state, the reward, and a "done" signal if the episode is complete.
 - The agent is then passed the tuple of (state, action, reward, next state, done signal) which it uses
   to update its memory and Q networks.
 - Epsilon is then decayed
 - The above is repeated until the environment returns a done signal or the maximum alloted number
   of episodes is reached.

##### DQN Agent

The DQN agent is a Python class which wraps two identical Pytorch neural network models. 
Each Q network was used as provided and has this structure:

```
Fully Connected Layer (64 units)
		  |
		ReLU
		  |
Fully Connected Layer (64 units)
		  |
		ReLU
		  |
Fully Connected Layer (4 units)
```

The networks were trained using the [Adam optimizer](https://pytorch.org/docs/stable/optim.html#torch.optim.Adam).
The learning rate (`LR`) was set to 0.0001. Betas, eps, weight_decay, and amsgrad were all left as 
default.

The first Q network (the "local" network) was used for action selection. The second
Q network was the target network and was used as a more stable reference during the temporal 
difference (TD) error calculation.

###### Action Selection

The current state was fed to the local Q network to obtain an array of action values. This
was then sampled in an epsilon-greedy fashion. Epsilon was initialized to 1.0 and decayed by 0.995
each episode. A minimum epsilon was set to 0.1 to preserve some exploration even late in training.
(i.e. episode 460 and beyond)

###### Updates and Learning

The agent maintained a replay buffer of up-to 10000 memories i.e. (state, action, reward, 
next_state, done) tuples. Every 4 steps the agent would sample 64 memories randomly from this 
buffer and use those to compute TD errors. A discount factor (`gamma`) of 0.99 was applied during
the calculation of TD errors.

Those errors were then used to compute a mean squared error (MSE) for the batch. This was passed to
Pytorch to calculate error gradients. Finally the network weights were updated by the Adam 
optimizer using those gradients.

After updating the local network (the network from which actions are chosen) a "soft" update was
applied to the target network such that it would track the learning of the local network but at
a greatly reduced rate. The fractional update was controlled by the parameter `tau` which was
set to 0.001.

Note that there was no prioritization of replay samples.

## Ideas for Future Work - Meet the Minimum Score Criteria

### Investigation

As noted above the trained agent demonstrates remarkably poor performance on *some* episodes even
after reaching a relatively high level of average performance. It would be interesting to explore
exactly why some episodes are so challenging to the agent. To do so developing a method to isolate
and replay those *episodes* would be useful.

### Prioritized Replay from Failed Episodes

While the original approach to prioritized replay relied on the TD error to determine
which state transitions were "useful" another metric could be to focus on entire episodes that
were challenging for the agent and prioritize learning from those. Extending this one might explore
if focusing on the extremes of performance, the very good and the very bad and prioritizing those
episodes for learning could help narrow the range of performance across episodes.

## Conclusion

In this project we adapted a stock DQN agent to a novel environment, providing useful adapter code
along the way. In addition we explored human level performance in this environment and used it
to contrast with the performance and robustness of a trained agent. As a consequence of that
exploration we discovered that agents which meet the original criteria for solving this
environment do not actually produce robust performance. In response we proposed an alternative
solution criteria which would better match human style play and better select for agents with
robust performance in this environment. Specifically that the Bananas environment is considered 
solved when the agent maintains a minimum score of 10 for 500 episodes.
