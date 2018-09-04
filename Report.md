# Implementing a DQN Agent

## Summary

 In this project I trained a DQN reinforcement learning agent to reach a score of +13 on 
 average over 100 episodes in the Unity-ML environment.  In this environment positive reward 
 is accumulated by running into yellow "good" bananas and  avoiding blue "bad" bananas which return -1 reward. 
 An episode ends after a fixed interval of 1000 steps.


I have implemented several variants of DQN Agents

# 1. DQN Agent with Experience Reply

DQN agent implements Deep Q-Learning Algorithm which has the following steps 

1. Initialize parameters for Q(s, a) and Q ˆ(s, a) with random weights, epsilon ← 1.0,
and empty replay buffer

2. With probability epsilon, select a random action a, otherwise a = arg max<sub>a</sub> Q <sub>s,a</sub>

3. Execute action in Unity-ML environment and observe reward r and the next state s′

4. Store transition (s, a, r, s′) in the replay buffer (buffer size=10000)

5. Sample a random minibatch (batch size = 64) of transitions from the replay buffer

6. For every transition in the buffer, calculate target y = r if the episode has
ended at this step or y = r + γ max<sub>a'∈A</sub>a' Q ˆ<sub>s',a'</sub> otherwise
A discount factor - gamma of 0.99 is applied.

7. Calculate loss: L = (Q<sub>s,a</sub> − y)<sup>2</sup>

8. Update Q(s, a) using the SGD algorithm by minimizing the loss in respect
to model parameters.  The learning rate for Adam optimizer (`LR`) was set to 0.0001. 
Betas, eps, weight_decay, and amsgrad were all left as default (this applies to all networks as described below).

9. Every 4 steps copy weights from local Q-network to target Q-network <sub>t</sub>
The target network is frozen for several time steps and then the target network weights are updated 
by copying the weights from the actual Q network. Freezing the target network for a while 
and then updating its weights with the actual Q network weights stabilizes the training.
The fractional update was controlled by the parameter tau which was set to 0.001.

10. Repeat from step 2 until reaching average reward over last 100 episodes of 13 


The Deep Q-Network architecture of the Basic DQN Agent is following (for vector observations):

```
Fully Connected Layer (in=37 -> state size, out=128)
		  |
		ReLU
		  |
Fully Connected Layer (in=128, out=128)
		  |
		ReLU
		  |
Fully Connected Layer (in=128, out=4 -> action size)
```

## 2. DQN Agent with Prioritized Experience Reply 

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

Snippet of the code of the explicit MSE Loss calculation:

    # Get max predicted Q values (for next states) from target model
    Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)

    # Compute Q targets for current states 
    Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

    # Get expected Q values from local model
    Q_expected = self.qnetwork_local(states).gather(1, actions)

    # Compute loss
    losses_v = weights * (Q_expected - Q_targets) ** 2
    loss = losses_v.mean()
    prios = losses_v + 1e-5
  
    # Minimize the loss
    self.optimizer.zero_grad()
    loss.backward()
    self.optimizer.step()

    # Update replay buffer priorities
    self.memory.update_priorities(idxes, prios.data.cpu().numpy())
  

The Deep Q-Network architecture of the DQN Agent is following (the same as for basic DQN Agent):

```
Fully Connected Layer (in=37 -> state size, out=128)
		  |
		ReLU
		  |
Fully Connected Layer (in=128, out=128)
		  |
		ReLU
		  |
Fully Connected Layer (in=128, out=4 -> action size)
```

## 3. DQN Agent for pixels navigation with Prioritized Experience Reply

DQN Agent implementation is identical to DQN Agent with Prioritized Experience Replay. 

What's different is the Q-Network.

In order to process visual observations I defined Q-Network as following:

```
2D Convolutional Layer (out=32, kernel size=8, stride=4)
		  |
		ReLU
		  |
2D Convolutional Layer (in=32, out=64, kernel size=8, stride=4)
		  |
		ReLU
		  |
2D Convolutional Layer (in=64, out=64, kernel size=8, stride=4)
		  |
		ReLU
		  |
Fully Connected Layer (in=64*7*7=3136 units, out=128)
		  |
		ReLU
		  |
Fully Connected Layer (in=128 units)
		  |
		ReLU
		  |
Fully Connected Layer (4 units - action size)
```
 
## 4. Dueling DQN Agent with Prioritized Experience Reply

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
the mean for advantage to zero: Q(s, a) = V (s) + A(s, a) - \frac{1}{k} \sum A(s, k)

To implement Dueling DQN Agent, its only necessary make changes to the Q-Network in the following way.

Value function network:

```
Fully Connected Layer (in=37 ->state size, out=128)
		  |
		ReLU
		  |
Fully Connected Layer (in=128, out=128)
		  |
		ReLU
		  |
Fully Connected Layer (in=128, out=1)
```

Advantage function network:

```
Fully Connected Layer (in=37 ->state size, out=128)
		  |
		ReLU
		  |
Fully Connected Layer (in=128, out=128)
		  |
		ReLU
		  |
Fully Connected Layer (in=128, out=4 -> actions size)
```
    def forward(self, x):
        val = self.fc_val(x)
        adv = self.fc_adv(x)
        return val + adv - adv.mean()

The Dueling DQN Agent implementation is the same as implementation of Prioritized DQN Agent, the difference is the Q-Network as described above.

## 5. Categorical DQN Agent 

The fundamental difference between all of the aboved described DQN Agents and Categorical DQN Agent 
is that Q-values in Categorical DQN Agent are replaced with more generic Q-value probability distribution.

Both the Q-learning and value iteration methods are working with the values of actions or states 
represented as simple numbers and showing how much total reward we can achieve from state or action. 
However, in complicated environments, the future could be stochastic, giving us different values with
different probabilities. 

The overall idea is to predict the distribution of value for every action.
As shown in the original paper the Bellman equation can be generalized
for a distribution case and it will have a form Z(x, a) = D R(x, a) + γZ(x', a'), which
is very similar to the familiar Bellman equation, but now Z(x, a), R(x, a) are the
probability distributions and not numbers.

The resulting distribution can be used to train our network to give better predictions
of value distribution for every action of the given state, exactly the same way as
with Q-learning. The only difference will be in the loss function, which now has to
be replaced to something suitable for distributions' comparison. There are several
alternatives available, for example Kullback-Leibler (KL)-divergence (or crossentropy loss) 
used in classification problems or the Wasserstein metric.

The main part of the method is probability distribution, which we're
approximating. There are lots of ways to represent the distribution, 
here i used generic parametric distribution that is basically a fixed
amount of values placed regularly on a values range. The range of values should
cover the range of possible accumulated discounted reward. In the paper, the
authors did experiments with various amounts of atoms, but the best results were
obtained with the range being split on N_ATOMS=51 intervals in the range of values
from Vmin=-10 to Vmax=10.

For every atom (we have 51 of them), the network predicts the probability that
future discounted value will fall into this atom's range. The central part of the
method is the code, which performs the contraction of distribution of the next
state's best action using gamma, adds local reward to the distribution and projects
the results back into our original atoms.

The architecture of the Q-Network is following:

```
Fully Connected Layer (in=37 -> state size, out=128)
		  |
		ReLU
		  |
Fully Connected Layer (in=128, out=128)
		  |
		ReLU
		  |
Fully Connected Layer (in=128, out=4*51 -> actions size*number of atoms)
```

As the Q-Network predicts probability distributions of actions forward method uses softmax:

    def forward(self, x):
        batch_size = x.size()[0]
        fc_out = self.fc(x)
        logits = fc_out.view(batch_size, -1, N_ATOMS)
        probs = nn.functional.softmax(logits, 2)
        return probs


### System Setup

 - Python:			3.6.6
 - CUDA:			9.0
 - Pytorch: 		0.4.1
 - NVIDIA Driver:  	390.77
 - Conda:			4.5.10

 All training was performed on a single Ubuntu 18.04 desktop with an NVIDIA GTX 1080ti. 

## Conclusion

Comparing the performance scores shows that Dueling DQN Agent and DQN Agent with Prioritized Experience Replay achieved the best performance. 

Navigation with Pixels - so far i have tested Basic DQN Agent with Prioritized Experience Replay using visual observations (frames). 
Unfortunately DQN Agent performance didnt show significant progress during the learning. 
I will try to implement few changes/improvements:
    frame stacking (buffering) 
    reward clipping

Categorical DQN - while this DQN agent good shows learning progress during training, i was expecting that the overall performance would exceed Dueling DQN agent 
and DQN Agent with Prioritized Experience Replay. I will try to investigate and test further. 

