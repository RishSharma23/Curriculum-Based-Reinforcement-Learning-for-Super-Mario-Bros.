# Curriculum-Based-Reinforcement-Learning-for-Super-Mario-Bros.
# Report

**Final Project Report for CS 175**

**Project Title**: Curriculum-Based Reinforcement Learning for Super Mario Bros.  
**Project Number**: 12

Student Name(s)  
Yoav Feigenbaum, 42336836, yfeigenb@uci.edu   
Rishit Sharma, 45329724, rishits3@uci.edu  
Eshaan Rawat, 33969625, erawat@uci.edu

**1\. Introduction and Problem Statement**  
	Reinforcement learning (RL) has been shown to be an effective methodology to train adaptive agents in simple environments with a limited action space and straightforward reward function. However, real-world environments are significantly more complex, and current RL algorithms often struggle to adapt. For our project, we explored one approach that aims to address the challenge of complex environments: progressive learning curriculums. We implemented several types of curriculums and evaluated their effectiveness compared to a traditional base agent on the classic platformer game Super Mario Bros. We chose this environment due to its large action space (Mario has 12 actions available which can be stringed into long sequences) and its complex reward function (the player is scored on multiple components such as progression, speed, and avoiding damage).   
	Our results demonstrate that training with curriculums significantly speeds up the learning process, allowing our agent to achieve better performance in less time. More interestingly, our experiments suggest that a curriculum applied only to the reward function is not as effective as a curriculum applied to the action space, yet the combination of the two significantly outperforms them individually.

**2\. Related Work**  
	Curriculum-based reinforcement learning has recently emerged as an active direction of research within both the RL community and in the wider field of machine learning. Published only a few months ago, the [paper](https://arxiv.org/pdf/2507.22607) “VL-Cogito: Progressive Curriculum Reinforcement Learning for Advanced Multimodal Reasoning” by Yuan et al. introduces a multimodal reasoning model that is trained through a progressive curriculum that gradually increases the difficulty of the reasoning tasks given to the learner. They show how this framework enables their model to surpass existing models on multimodal reasoning benchmarks. Although these results suggest that progressive curriculums are a compelling approach for learning complex tasks, they do not systematically evaluate which configurations of curriculum components and parameters are the most effective, which is the question we explore in our project.   
	Additionally, since our project is focused on the implementation of different types of curriculums and performing experiments, we relied on existing work for the environment and parts of the baseline agent. The Super Mario Bros. environment implementation we used, which can be found [here](https://github.com/Kautenja/gym-super-mario-bros), employs a Nintendo Entertainment System (NES) emulator to construct a gym environment that faithfully recreates the original game. We also used Stable Baselines 3 for the Proximal Policy Optimization (PPO) algorithm, which our agent is built on top of. Taking advantage of these existing resources allowed us to focus our efforts into the novel component of our project: evaluating curriculums. 

**3\. Data Sets**  
	Unlike supervised learning, which relies on training and testing data, our Reinforcement Learning approach does not use external data sets. *Instead,* both the baseline and curriculum agents learn through direct interaction with the Super Mario Bros. gym environment. The PPO algorithm uses state-action-reward tuples from each training episode to update the policy network.    
	The provided Super Mario Bros. gym environment serves as our primary data source, providing the agent with visual observations (game frames), action feedback, and reward signals at each timestep. The state space consists of 240×256 pixel RGB frames captured from the NES emulator.   
For both of our agents, we focused on Level 1-1 and Level 1-2 of Super Mario Bros. These levels were selected because they represent the introductory stages of the game, offering a manageable complexity for systematic curriculum evaluation while still presenting meaningful challenges such as gaps, pipes, Goombas, and Koopa Troopas. World 1-1 is an outdoor grassland level with relatively straightforward platforming, while World 1-2 is an underground level with darker visual characteristics and different obstacle configurations. By training on two visually different levels, we ensure that our agents learn generalizable policies and do not overfit.   
<img width="298" height="275" alt="image" src="https://github.com/user-attachments/assets/c2eedd43-72a4-4327-913b-688e7b9db73c" />
<img width="298" height="275" alt="image" src="https://github.com/user-attachments/assets/3631ba0d-c546-4cbe-90ac-00bcb3d1022e" />
*(Left) World 1-1, the classic outdoor grassland level used for training and evaluation.*  
*(Right) World 1-2, the underground level providing visual and structural variety for agent training.*

**4\. Description of Technical Approach**

**Overview**: Our project was comprised of three major components: 

1. Training a baseline agent on the Super Mario Bros. gym environment  
2. Implementing multiple curriculum types with easily accessible parameters  
3. Evaluating the different configurations in order to draw conclusions

This section will describe the technical approaches used for (1) and (2). See the section on Experiments and Evaluation for more information on (3). 

**Baseline Agent**: We began with implementing a traditional reinforcement learning agent that is able to play Super Mario Bros. The agent is trained using the Proximal Policy Optimization (PPO) algorithm, which is considered state-of-the-art for policy-based reinforcement learning. We chose to use a policy-based approach rather than a value-based approach because Super Mario Bros. has an immensely large state space, nonlinear rewards, and it requires sequences of actions, which are handled better by policy-based methods. For the implementation of PPO, we used Stable Baselines 3, specifically utilizing the built-in convolutional neural network (CNN) policy in order to extract features from the state representation, which in this case is an image.  
Since the existing Super Mario Bros. gym environment (see Related Work) is incompatible with gymnasium, the maintained version of gym, which is deprecated, we created several wrappers that allow us to work with gymnasium, making dependency management more straightforward (although this was still one of the most challenging parts of our project). These wrappers also ensure that the state representation is resized appropriately and converted to grayscale to fit what Stable Baselines 3’s CNN expects.  
	  
**Curriculums**: To create a flexible progressive curriculum framework that can be easily configured for experimentation, we wrote our own subclass of the Super Mario Bros. environment. Our modifications target two specific components: reward calculation and action selection. Additionally, we track the number of training steps that have been completed and progress to the next stage of the curriculum when appropriate. The specific curriculum of reward function progression and action space expansion to follow, as well as the intervals between stages, are passed to the environment as parameters.  
	For the replacement of the reward function, we return a combination of reward components that is dependent on the current stage of the curriculum. The possible components that can be included are a reward for rightward progression towards the goal, a penalty for dying, and a penalty for stalling the game. An example curriculum could be training with only the progression component for 100000 steps, then using progression and death for another 200000 steps, and finally, using all three for the remainder of training. The default, which is used for the base agent, is including all three components for the entirety of training. Our experiments explore how the order of progression, as well as the interval sizes, affect the learning process.  
	Similarly, we override the action selection mechanism in order to allow for a progressive curriculum in the action space. At every curriculum stage, a subset of the total action space is defined as allowed, and the forbidden actions are remapped to to have the same effect as no action. We chose to remap forbidden actions rather than modify the size of the policy network’s output layer in order to prevent harsh network resets during training. Although any subset of the action space can be defined as a curriculum stage, some intuitive subsets we used for our experiments included right-only actions and no diagonal jumping. As with the reward function modifications, the base agent trains with the complete action space.

**5\. Software**

**Open Source Software:**

Languages: Python  
Libraries: numpy, gymnasium, pandas, matplotlib, PyTorch, Stable Baselines3, Other Dependencies (listed in requirements.txt file)  
Tools: Anaconda (Environment Setup), Jupyter Notebook

**Super Mario Gym Environment**: Our group used this environment as the basis for training our RL agents. The documentation for that can be found [here](https://pypi.org/project/gym-super-mario-bros/). Similar to previous homework assignments, this allowed our group to easily model the available action space and reward function to train our PPO algorithm.

**Baseline Agent:** Our group referenced existing open source software to compare the differences between a baseline PPO agent (full action space and reward function available) and our implemented curriculum based learning agent. Specifically, we used:

- COMPLEX\_MOVEMENT action set defined in gym\_super\_mario\_bros/[actions.py](http://actions.py).  
- JoypadSpace wrapper from nes\_py/wrappers/joypad\_space.py to map the native 256-action NES space to this 12-action subset   
- The environment’s default reward function from gym\_super\_mario\_bros/smb\_env.py, r=v+c+d with reward clipping to \[−15,15\].  
- We also relied on the environment’s built-in info dictionary from *smb\_env.py*.   
- Level ID’s from *gym\_super\_mario\_bros/\_registration.py*

The code for the baseline agent can be found in this [GitHub repository](https://github.com/vietnh1009/Super-mario-bros-PPO-pytorch). 

**Original Software:**

**Curriculum Agent:** Our group used original code to develop a curriculum based agent, capable of incrementally increasing the search space and the complexity of the reward function as the number of timesteps increases. We did this by modifying a custom wrapper class around the pre-existing gym environment to dynamically set the parameters for the actions and rewards.

**Experimentation Framework:** We also created a training and evaluation pipeline that enabled us to run several experiments and analyze the results.

**6\. Experiments and Evaluation** 

**Experiment Setup**: Our experiment involved comparing the baseline agent to our curriculum based learning agent to determine whether or not a curriculum based approach would speed up convergence time over N timesteps. For reproducibility, our group decided on 1 million (1M) iterations as it provided a good balance between rigorous training time for a good portion of the level without producing an exponentially high computational cost normally associated with 5-10M timesteps.  
We would then compare both agents on two sample words (1-1 and 1-2) to determine their performance. Our group had multiple variations of a baseline agent, where we varied the curriculum according to the below process:  
Agent A \- Full action space available, test on incremental rewards  
Agent B \- Incremental action space, full reward function available  
Agent C \- Incremental action spaces and reward functions 

**Evaluation:** The agents were evaluated on two main criteria. The first was the average mean reward over 1M timesteps, which was computed through the reward function, a combination of distance traveled, time efficiency, and not dying to enemies. The second criteria was level completion rate, measured in number of flags (fractional) which signified how close the agent was to solving a given level.

The baseline agent had the following results on levels 1-1 and 1-2 respectively:  
<img width="631" height="437" alt="image" src="https://github.com/user-attachments/assets/7454edbd-5cf6-4f40-9618-c52324c2ff50" />


The full curriculum agent (Agent C) had  the following results on levels 1-1 and 1-2 respectively:  
<img width="627" height="222" alt="image" src="https://github.com/user-attachments/assets/b8d8ab43-caa7-479e-b73e-eb8549d241e6" />
<img width="627" height="222" alt="image" src="https://github.com/user-attachments/assets/01854857-3aee-4cbf-b7bb-ce21697db6e9" />


**Results**: On average, the curriculum agent displayed a higher mean reward as well as a faster time to converge to an optimal policy, as shown on the green graphs on the right hand side. The most notable difference occurred on level 1-2, where the curriculum based agent had a much higher average reward of 700+ compared to the baseline agent of only 500, where the baseline agent also failed to converge. Additional results have been included in the below appendix, with labeled figures.

**7\. Discussion and Conclusion** 

As indicated by our results, training a reinforcement learning agent with a progressive learning curriculum significantly speeds up the learning process, achieving better performance early on. This aligned with our intuition, but the experimental data serves as validation. Although we expected the reward function curriculum to be the primary contributing factor, our ablation study suggests that the action space curriculum is more effective, at least when using them individually. Another interesting finding is that combining the two curriculum frameworks has a constructive effect, allowing the agent to achieve much performance better than only using either of the two components.  
One aspect that would be intriguing to explore in the future is the effect of the curriculum interval lengths on the training process. The code we wrote allows this parameter to be modified, so testing this would be relatively straightforward and could yield insights into how progressive curriculums should be structured to achieve optimal performance. Additionally, we could explore other training components that a curriculum could be used towards, such as the state space.  
All in all, this project provided us the opportunity to engage in a research project that has deepened our understanding of reinforcement learning and its applicability to a broad range of settings.

**8\. Individual Contributions** 

**Eshaan Rawat**: I was responsible for running each experiment and modifying the software to support visualizations of the results. While Yoav and Rish built out their respective agents, my work focused on running training iterations for multiple curriculum agents, saving and loading each model, as well as fine tuning hyperparameters. Initially, I experimented with different hyperparameters such as the learning rate, discount factor, and the clipping parameter for PPO, which was a slightly different process for the baseline agent and the curriculum based agent. 

After choosing appropriate variables, I then focused on running each model for 1M timesteps on levels 1-1 and 1-2, and plotting the results as mentioned above in our analysis. This step also involved code for saving the models in case we wanted to re-run / re-evaluate them, and also code to visualize the agent playing frame by frame. Finally, I consolidated the results on a spreadsheet and compared which agents were performing better, noting the strengths and weaknesses of certain agents, such as the incremental learning agent, the incremental action agent, and the fully incremental curriculum based agent.

**Yoav Feigenbaum**: My primary responsibility was to design and implement the curriculum-based agent, which would allow for an intuitive experimentation pipeline. To this end, I wrote the code for the Super Mario Bros. environment subclass, which overrides several methods, including the reward function and action selection, as well as adding new methods to keep track of the current stage of the curriculums.   
Additionally, I selected components of the reward function that could be included in a curriculum, and set the default behavior (for the base agent) to use all of them. Similarly, I implemented the mechanics of action remapping during specified stages of a curriculum, and created a few subsets of actions that would be used as curriculum stages. The default base agent uses the full action space from the start.  
Finally, I designed the curriculum configurations to be experimented with, choosing a set of curriculums that would allow us to compare the effectiveness of the two curriculum types both individually and in conjunction with each other.

**Rish Sharma:**   
	My primary responsibility was developing the baseline PPO agent and establishing the foundational infrastructure that both agents would build upon. I began by researching the gym-super-mario-bros environment.  
	I then set up the initial Python notebook and configured the conda environment for our experiments. One of the most significant challenges I encountered was resolving dependency conflicts between the gym-super-mario-bros library, which relies on the deprecated OpenAI Gym API, and Stable Baselines 3, which expects the newer Gymnasium interface. To address this incompatibility, I developed a series of custom wrapper classes, including a GymnasiumCompatibilityWrapper that translates between the two APIs, ensuring proper handling of the different return formats from the step and reset functions.  
I also implemented preprocessing wrappers for frame skipping, grayscale conversion, and observation resizing to prepare the raw game frames for input into the CNN policy network. Beyond the environment setup, I wrote the complete baseline agent code, including the training loop, a CNN feature extractor, and various training callbacks for logging metrics and saving checkpoints. I also addressed issues such as policy collapse by tuning the entropy coefficient and implemented functionality for tracking episode statistics like maximum x-position reached and flag completion.  
This baseline implementation served as the control condition, and the infrastructure I built was reused by Yoav when developing the curriculum agent.

**Appendix**

**Figure 1.1**: Result of Agent A (full action space available, test on incremental reward function) after training for 5M timesteps on level 1-1.  
<img width="624" height="213" alt="image" src="https://github.com/user-attachments/assets/fee1ae86-cdc0-438b-8d02-a96cd3b95350" />

**Figure 1.2**: Result of Agent A (full action space available, test on incremental reward function) after training for 1M timesteps on level 1-1.  
<img width="612" height="210" alt="image" src="https://github.com/user-attachments/assets/54dbdfd8-4e21-4935-ae6f-78f974dbea9b" /> 

**Figure 1.3**: Result of Agent B (incremental action space, full reward function available) after training for 1M timesteps on level 1-1.  
<img width="627" height="212" alt="image" src="https://github.com/user-attachments/assets/3c19b13f-70f0-4818-90f9-c9df563100ad" /> 

**Figure 1.4**: Result of Agent C (incremental action spaces and reward functions) after training for 1M timesteps on level 1-1.  
<img width="626" height="214" alt="image" src="https://github.com/user-attachments/assets/47ad38b1-5207-41e3-a5d4-e9d6f3b0da78" />


