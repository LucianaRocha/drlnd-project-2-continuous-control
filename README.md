[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/43851024-320ba930-9aff-11e8-8493-ee547c6af349.gif "Trained Agent"
[image2]: https://user-images.githubusercontent.com/10624937/43851646-d899bf20-9b00-11e8-858c-29b5c2c94ccc.png "Crawler"
[image3]: ./images/Navigate.png "Navigate"
[image4]: ./images/Environment.png "Environment"
[image5]: ./images/Run.png "Run"


# Project 2: Continuous Control

## Project Details

In this project, I worked with the [Reacher](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#reacher) environment.

![Trained Agent][image1]

In this environment, a double-jointed arm can move to target locations. A reward of +0.1 is provided for each step that the agent's hand is in the goal location. Thus, the goal of agent is to maintain its position at the target location for as many time steps as possible.

The observation space consists of 33 variables corresponding to position, rotation, velocity, and angular velocities of the arm. Each action is a vector with four numbers, corresponding to torque applicable to two joints. Every entry in the action vector should be a number between -1 and 1.

### Distributed Training

For this project, [Udacity](https://www.udacity.com/) provides us with two versions of the environment:
  
- The first version contains a single agent.  
- The second version contains 20 identical agents, each with its own copy of the environment.  

The second version is useful for algorithms like [PPO](https://arxiv.org/pdf/1707.06347.pdf), [A3C](https://arxiv.org/pdf/1602.01783.pdf), and [D4PG](https://openreview.net/pdf?id=SyZipzbCb) that use multiple (non-interacting, parallel) copies of the same agent to distribute the task of gathering experience.  

### Solving the Environment  
In my project submission I needed only solve one of the two versions of the environment:

**Option 1: Solve the First Version**  
The task is episodic, and in order to solve the environment,  your agent must get an average score of +30 over 100 consecutive episodes.

**Option 2: Solve the Second Version**  
The barrier for solving the second version of the environment is slightly different, to take into account the presence of many agents.  In particular, your agents must get an average score of +30 (over 100 consecutive episodes, and over all agents).  Specifically,  

- After each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent.  This yields 20 (potentially different) scores.  We then take the average of these 20 scores.  
- This yields an **average score** for each episode (where the average is over all 20 agents).

**_For my project I chosen the Second Version._**   
The environment is considered solved, when the average (over 100 episodes) of those average scores is at least +30. 

## Getting Started

### Files included in this repository  

- The files below are needed to create and to train the Agent:
    - Continuous_Control.ipynb  
    - ddpg_agent.py  
    - model.py 
    - _old-model.py (along of my tests I used different models, this was the first model)_

- The trained model:
    - checkpoint.pth

- Useful information about the project:
    - README.md

- File describing the development process and the learning algorithm:
    - Report.md  

## Preparing the environment to run the project  

**1.** Install the Anaconda Platform

- For more information about it, see [Anaconda Installation Guide](https://docs.anaconda.com/anaconda/install/)
- Download the [Anaconda Installer](https://www.anaconda.com/distribution/), version Python 3.x

**2.** Create and activate a new environment for this project

- Open Anaconda Prompt (Windows) or Terminal (Linux) and type

```
conda create --name continuous python=3.6 -y
```  

```
conda activate continuous
```

**3.** Install Pytorch  
Numpy and other required libraries will be installed with Pytorch.

 - In your Anaconda Prompt (Windows) or Terminal (Linux) type the line command below

```
conda install pytorch -c pytorch -y
```
**4.** Install Unity Agents  
Matplotlib, Jupyter, TensorFlow, and other required libraries will be installed with Unity Agents.

 - In your Anaconda Prompt (Windows) or Terminal (Linux) type the line command below
 
```
pip install unityagents
```


## Getting the code
There are two options to get this project:

**1.** Download it as a zip file  

 - Do the download in this [link](https://github.com/LucianaRocha/drlnd-project-2-continuous-control/archive/master.zip).  
 - Unzip the file in a folder of your choice.

**2.** Clone this repository using Git version control system 
 
 - Open Anaconda Prompt (Windows) or Terminal (Linux), navigate to the folder of your choice, and type the command below:  

```
git clone https://github.com/LucianaRocha/drlnd-project-2-continuous-control.git
```

 - If you want to know more about Git, see this [link](https://git-scm.com/downloads).

## Download the environment 
 
Download the environment, unzip (or decompress) the file, and place it inside the folder drlnd-project-2-continuous-control, where you downloaded or cloned the repository. You need only select the environment that matches your operating system:

- **_Version 1: One (1) Agent_**  
	- Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Linux.zip)  
	- Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher.app.zip)  
	- Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Windows_x86.zip)  
	- Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Windows_x86_64.zip)  

- **_Version 2: Twenty (20) Agents_**  
	- Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux.zip)  
	- Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher.app.zip)  
	- Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86.zip)  
	- Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86_64.zip)  
    
    (_For Windows users_) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

    (_For AWS_) If you'd like to train the agent on AWS (and have not [enabled a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md)), then please use [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Linux_NoVis.zip) (version 1) or [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux_NoVis.zip) (version 2) to obtain the "headless" version of the environment.  You will **not** be able to watch the agent without enabling a virtual screen, but you will be able to train the agent.  (_To watch the agent, you should follow the instructions to [enable a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md), and then download the environment for the **Linux** operating system above._)



## Running the project

- Open Anaconda Prompt (Windows) or Terminal (Linux), navigate to the project folder, and type the commands below to activate the project environment, and to open Jupyter.  
If you are keen to know more about notebooks and other tools of Project Jupyter, you find more information on this [website](https://jupyter.org/index.html).  

```
conda activate continuous 
```

```
jupyter notebook  
```

- Click on the Continuous_Control.ipynb to open the notebook. 

![Navigate][image3]

- Next, we will start the environment! Before running the code cell below, change the file_name parameter to match the location of the Unity environment that you downloaded.

![Environment][image4]

- Run the notebook:

![Run][image5]
  


