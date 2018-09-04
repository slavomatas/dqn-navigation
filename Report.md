# Implementing a Banana Obsessed Agent

## Summary

 In this project I trained a DQN reinforcement learning agent to reach a score of +13 on 
 average over 100 episodes in the Udacity Deep Reinforcement Learing Nanodegree Bananas 
 environment. (A simplified version of the Banana Collectors Unity-ML environment.) 
 In this environment positive reward is accumulated by running into yellow "good" bananas and 
 avoiding blue "bad" bananas which return -1 reward. An episode ends after a fixed interval of 300 
 steps.


I have implemented several variants of DQN Agents and Deep Q-Networks

## 1. Basic DQN Agent with replay buffer

Basic DQN agent implements Deep Q-Learning Algorithm which has the following steps 

1. Initialize parameters for Q(s, a) and Q ˆ(s, a) with random weights, \epsilon ← 1.0,
and empty replay buffer
2. With probability \(\epsilon\), select a random action a, otherwise a = arg max<sub>a</sub> Q <sub>s,a</sub>
3. Execute action a in ML Agents and observe reward r and the next state s′
4. Store transition (s, a, r, s′) in the replay buffer
5. Sample a random minibatch of transitions from the replay buffer
6. For every transition in the buffer, calculate target y = r if the episode has
ended at this step or y = r + γ max<sub>a'∈A</sub>a' Q ˆ<sub>s',a'</sub> otherwise
7. Calculate loss: L = (Q<sub>s,a</sub> − y)<sup>2</sup>
8. Update Q(s, a) using the SGD algorithm by minimizing the loss in respect
to model parameters
9. Every 4 steps copy weights from Q to Q ˆ<sub>t</sub>
10. Repeat from step 2 until reaching average reward over last 100 episodes of 13 


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
  

## 3. DQN Agent for pixels navigation with prioritized replay buffer
 
## 4. Dueling DQN Agent

## 5. Categorical DQN Agent 


### Learning

 The `freeplay.py` script provides a human interface to the environment. Using the `wasd` keys
 a user can control the agent and attempt to collect reward. (Note: This is a Windows only script)

#### Complete Control

 The author made an initial attempt to learn how to play the game by directly controlling all 
 aspects of the agents movement. (i.e. `forward`, `backward`, `left turn`, `right turn`) Over 20 
 episodes (roughly 30 minutes) the author achieved a maximum score of +12. 

 There were two major limitations in this approach. First, the implementation of our key capture 
 didn't handle switching between long-held keys well. This meant that holding forward and turning 
 left and right as needed (a common play style for first-person movement) was jerky and unreliable. 
 Second, the all-or-nothing (discrete) nature of turns made gameplay more challenging than expected. 
 It should be noted that this limitation is unique to the Udacity Deep Reinforcement Learning 
 Nanodegree Bananas environment and does not appear in the Unity-ML Banana Collectors environment 
 where the action space is continuous.

#### Modified Control

 A slight modification was made to `freeplay.py` to have the default action be `forward` which leaves
 the user free to concentrate on only `left turn`, `right turn` and `backward`. In practice only
 `left turn` and `right turn` were used. After only 5 minutes of additional play time 
 (4 episodes) with this new control scheme the author's max score rose to +19.

 Thus in a total of 24 episodes the author was able to greatly exceed the threshold of +13 for 
 "solving" this environment. 

 A second person (human 2) was recruited to learn how to play Bananas to compare their learning 
 rate to the author's. They started with the modified control scheme and were given instruction on the control
 scheme and the goal of the game.

<p align=center>
	<img width=70% src="images/human-2-learning.png"/>
</p>

 They played 12 consecutive episodes and their scores were recorded. After 4 episodes
 they scored +17. They averaged +12.55 over the 12 episodes.

 But what about average performance over time?

#### 100 Episode Average was +16 with a minimum of +12

 To determine the longer term performance characteristics of a human player the author then played 
 100 consecutive episodes. As can be seen in the below chart there is significant variation between 
 episodes with a min score of +12 (N=2) and a maximum score of +22 (N=2). The standard
 deviation was 2.48. 

<p align=center>
	<img width=70% src="images/human-performance.png"/>
</p> 

 The environment has a random component where the placement of good and bad bananas is stochastic 
 as is the placement and orientation of the player/agent at the start of each episode. This results
 in favorable and unfavorable configurations of the play field for the player. 

 A favorable configuration might have yellow bananas cleanly segregated from blue bananas and in a 
 tight clump making them easy to collect quickly. An unfavorable configuration might have bananas 
 evenly distributed across the play field with yellow bananas placed close to blue ones. This would
 maximize the travel time required between bananas and increase the chances of accidentally
 collecting a blue banana and thus lowering the total score.

## Agent Performance - Plot of Rewards

Over 16 trials with seeds randomly selected for each trial the DQN agent was able to reach an
average score of +13 over the previous 100 episodes in __460__ episodes on average. (Standard 
Deviation: 70.32). Below you can see the per-episode score of one these agents.

<p align=center>
	<img width=70% src="images/solve-avg-13.png"/>
</p>

### Reviewing Performance

After having reached the +13 score criteria `train.py` saves a checkpoint file to disk which
contains the trained weights of the component Q networks for the agent. Since training is done
without graphical output it is useful to review the actual performance of the agent visually.

Upon review it was noted that the agent was frequently failing to get a score of even +1. Because
the agent was trained with a different seed for the Bananas environment than is used during
review this suggests there might be some overfitting to the parameters of the training environment.
This however seemed unlikely due to the relatively small capacity of the network. Instead of
diving into the network structure we decided to investigate whether the performance of an agent
was robust even within a single environment seed regime.

### Reaching a Minimum Score of +10

If an agent has on-going robust performance within an environment its minimum score should improve
along with the average score. Eventually truly poor performance should be almost completely
eliminated. A pro-golfer may have bad games but they are never going to double-bogey every hole
as might an amateur.

<p align=center>
	<img width=70% src="images/solve-min-10-1295.png"/>
</p>

The above chart shows that this is only weakly the case. By modifying the training script to
terminate when the agent had a __minimum__ score of +10 over the last 100 episodes we were able
to gather data on more 'robust' performance. Two trials were run in this mode and the average
episodes to the "solve" criteria was __1543__ trials.

While this was substantially more than the number of trials required to reach the original
criteria it still didn't feel like the agent had become truly robust. To exhaustively test
this hypothesis an agent was allowed to train for unlimited episodes over night.

### Continued Failure

After approximately 10 hours of training time the agent completed 18000 episodes. For
each 100 episode period the minimum score was found and charted below.

<p align=center>
	<img width=70% src="images/continued-failure.png"/>
</p>

Surprisingly, even after 39 times the number of episodes required to meet the original solution 
criteria the agent had still not mastered the environment. At no point did the agent appear to be 
able to avoid the occasional catastrophic performance. Though the agent twice had 100 episode
periods where its minimum score was 11 it never got above that and regularly returned to low
minimums.

## Alternate Solution Criteria - Minimum Values

The original criteria for solving the Bananas environment is to reach a score of +13 on average
over the last 100 episodes. Unfortunately, occasional high scores mask the failure of the agent
to attain robust performance. After significant exploration of the long term learning
performance of the stock DQN agent it was determined that at no point did the agent consistently
avoid reverting to near-untrained scores in some episodes. To address the gap between what the
original criteria suggested was a solution to the environment and what our testing has shown we 
propose the following simple modification.

> The Bananas environment is considered solved when the agent maintains a minimum score of 10
> for 500 episodes.

These values were chosen to fall outside of the performance range demonstrated by the stock
DQN agent but within a reasonable extrapolation from peak agent and human level performance. Over 
many trials the maximum score observed from an agent was +30, far exceeding the maximum score 
observed in 100 episodes of human performance. Given that this score is attainable by an agent a 
value 1/3 of that maximum seems reasonable for a minimum bar, the minimum of +12 over 100 trials
for a human lends additional credence to the attainability of this value.

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
