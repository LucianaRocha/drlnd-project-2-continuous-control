[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/43851024-320ba930-9aff-11e8-8493-ee547c6af349.gif "Arm"
[image2]: ./images/a.png "Packages"
[image3]: ./images/q_table_example.png "MDP"
[image4]: ./images/network.png "Net"
[image5]: ./images/Rewards.png "Rewards"
[image6]: ./images/Hyperparameters.png "Hyper"
[image7]: ./images/loss_function.png "Loss"
[image8]: ./images/Hyper_dqn.png "Hyper_dqn"
[image9]: ./images/Actor_Critic.png "actor-critic"
[image10]: ./images/First_model.png "First_model"
[image11]: ./images/Second_model.png "Second_model"
[image12]: ./images/Plot_1_agent.png "Plot_1_agent"
[image13]: ./images/Plot_20_agents.png "Plot_20_agents"
[image14]: ./images/environment_20_agents.png "environment_20_agents"
[image15]: ./images/environment_1_agent.png "environment_1_agent"
[image16]: ./images/Final_H.png "Final_H"


# Project 2: Continuous Control
![Arm][image1]

## Glossary
This glossary may help the reader to understand the concepts used in this implementation. I collected these concepts in [Unity Tecnologies - Github](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Glossary.md) and in the [Deep Reinforcement Learning Nanodegree Program](https://www.udacity.com/course/deep-reinforcement-learning-nanodegree--nd893).

- **Action:** The carrying-out of a decision on the part of an agent within the environment.  
- **Agent:** Component which produces observations and takes actions in the environment. Agents actions are determined by decisions produced by a Policy.  
- **Decision:** The specification produced by a Policy for an action to be carried out given an observation.  
- **Environment:** The scene which contains Agents and the Academy.  
- **Observation:** Partial information describing the state of the environment available to a given agent. (e.g. Vector, Visual, Text)  
- **Policy:** Function for producing decisions from observations.  
- **Reward:** Signal provided at every step used to indicate desirability of an agent’s action within the current state of the environment.  
- **State:** The underlying properties of the environment (including all agents within it) at a given time.  
- **Trainer:** Python class which is responsible for training a given group of Agents.  

## Introduction
To develop this solution I used the concepts learned in the [DDPG paper](https://arxiv.org/abs/1509.02971), and in the [Udacity Deep Reinforcement Learning Nanodegree](https://www.udacity.com/course/deep-reinforcement-learning-nanodegree--nd893) course.

**Actor-critic methods** are at the intersection of value-based methods such as DQN and policy-based methods such as reinforce.  

![actor-critic][image9]

- A value-based agent uses a deep neural network to approximate a value function. It learns about the optimal action-value function.  
- A policy-based agent uses a deep neural network to approximate a policy. 

**An actor-critic agent** uses function approximation to learn a policy any value function.
A basic actor-critic agent uses two networks:  
 
- One network, the actor, takes in a state and outputs the distribution over actions.  
- The other network, the critic takes in a state and outputs a state-value function of policy π. It will learn to evaluate the state-value function π using the TD estimate. The critic calculates the advantage function and trains the actor using this value as a baseline. Actor-critic methods use value-based techniques to further reduce the variance of policy-based methods.

The most popular actor-critic agents to date are:  

- Asynchronous Advantage Actor-Critic (A3C). Explore the [Q-prop paper](https://arxiv.org/abs/1611.02247).  
- Advantage Actor-Critic (A2C). 
- Generalized Advantage Estimation (GAE). See the [GAE paper](https://arxiv.org/abs/1506.02438)
- Deep Deterministic Policy Gradient (DDPG). More information in the [DDPG paper](https://arxiv.org/abs/1509.02971).  



## Implementation 

**To solve this project I implemented an agent based in the [DDPG paper](https://arxiv.org/abs/1509.02971)**. Below I describe model architectures _(first model and second model)_, learning algorithms, and  hyperparameters.  
Furthermore, I describe the results obtained with each hyperparameter set tested.

### The training phase

To solve this project I began with a very simple model **_First Model_** that did not work very well. Then I built another model **_Second Model_** that I improved based in learned lessons in the [Udacity Deep Reinforcement Learning Nanodegree](https://www.udacity.com/course/deep-reinforcement-learning-nanodegree--nd893) course, and in the concepts presented in the [DDPG paper](https://arxiv.org/abs/1509.02971). Below I describe each one models.

**Model architecture**

The **Actor Network** receives as input 33 variables representing the observation space and generates as output 4 numbers representing the predicted best action for that observed state. That means, the Actor is used to approximate the optimal policy π deterministically.

The **Critic Network** receives as input 33 variables representing the observation space. The result of the Critic's first hidden layer and the action proceeding from the Actor Network are stacked to be passed in as input for the Critic's second hidden layer.
The output of this network is the prediction of the target value based on the given state and the estimated best action.
That means, the Critic calculates the optimal action-value function Q(s, a) by using the Actor's best-believed action.

- **First Model**  
Initially, I created the Actor and Critic networks each of them with two fully-connected hidden layers with ReLU activation and one fully-connected linear output layer. I defined my network as having an input of 33 variables, 128 nodes for the first hidden layer, 32 nodes for the second one.


- **Second Model**  
I built the layout suggested in the [DDPG paper](https://arxiv.org/abs/1509.02971): two hidden layers, the first with 400 nodes and the second with 300 nodes, for both Actor and Critic networks.


### Hyperparameters
My next step was a tuning phase on the hyperparameters. Within this phase, I also revisited my implementations to verify if I had written the code correctly.  
It's a difficult task since a minimal change in one hyperparameter can lead to a significant change in the final result. 

For the two models I followed three tips given by Udacity:  

	- Keep Gamma high
	- Keep learning rate low
	- Avoid setting epsilon to zero

For all the tests I ran 100 episodes and I used the average score to decide the better parameter.

- **First Model** 

	- fc_layers for the actor network: FC1: 128 nodes, FC2: 32 nodes
	- fc_layers for the critic network: FC1: 128 nodes, FC2: 32 nodes 

![First_model][image10]


- **Second Model**  

	- fc_layers for the actor network: FC1: 400 nodes, FC2: 300 nodes
	- fc_layers for the critic network: FC1: 400 nodes, FC2: 300 nodes

Along with the test phase, I implemented the items below:  

	- Ornstein-Uhlenbeck noise process (see the ddpg_agent.py)
	- Apply Sigmoid to the critic output (see the model)
	- Use Batch normalization (see the model)
	- Limit the iterations in the training loop (notebook)


When I achieved a good average score **_(run 7)_** I ran the project in GPU. This changing worsted the average score **_(run 8)_** so I continued improving my model (now in GPU). See the results below.


![Second_model][image11]  


In my final model I used the hyperparameters below, found in the **_run 16_**.  

![Final_H][image16]  

## Plot of Rewards
### Environment with 1 agent (Option 1)
This graph shows the rewards per episode for the agent within the training phase, as well as the moving average.
It illustrates that the Agent was able to receive a moving average reward of at least 30.0 points over 100 episodes. 

![Plot_1_agent][image15]
![Plot_1_agent][image12]


### Environment with 20 agents (Option 2)
After achieving a good result with my model in the environment with 1 agent, I ran the same project in the **environment with 20 agents.**

![environment_20_agents][image14]  

![Plot_20_agents][image13]

## Conclusion  

To solve this challenge, I studied and implemented the architecture proposed in the DDPG paper.  Following an extensive fine-tuning phase, I reached a truly impressive result, solving the environment with 20 agents in as few as possible episodes, requiring 0 extra episodes to do that.
Working on projects like this one involves a great deal of reading, studying,  attention,  and patience, especially in the tailoring of the hyperparameters.


## Ideas for Future Work
There are other algorithms proposed to solve this kind of environment. One future work could implement them to verify their performance in this environment. Those algorithms are:


- **TRPO:** [Trust Region Policy Optimization](https://arxiv.org/abs/1502.05477)
- **GAE:** [Generalized Advantage Estimation](https://arxiv.org/abs/1506.02438)
- **A3C:** [Asynchronous Advantage Actor-Critic](https://arxiv.org/abs/1602.01783)
- **A2C:** Advantage Actor-Critic
- **ACER:** [Actor Critic with Experience Replay](https://arxiv.org/abs/1611.01224)
- **PPO:** [Proximal Policy Optimization](https://arxiv.org/pdf/1707.06347.pdf)
- **D4PG:** [Distributed Distributional Deterministic Policy Gradients](https://arxiv.org/pdf/1804.08617.pdf)