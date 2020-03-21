# Baxter learns to lie a knot
Getting a robot to learn based on its interactions with environment is one of the most powerful ideas of all. 
Most of our object manipulation revolves around the rigid body assumption. But in this project, I implemented a work that enables 
the robot to learn to tie a knot purely based on interacting with its environment. This a self supervised deep learning technique 
applied to the reinforcement learning problem of learning an effective policy.w. This 
repository contains codes for an experiment, where Baxter learns to lie a knot. This
series of work was based on the principle of computational sensorimotor learning actively
followed by Prof. Jitendra Malik's group at UC, Berkely. More details about the work can be found in 
[this link](https://pathak22.github.io/zeroshot-imitation/). More details about my implementation
can be found at my [portfolio](https://senthilpalanisamy.github.io./)

# Library requirements
The library requirements are listed in each separate subdirectory

# Directory structure.
1. baxter_ros - A baxter package that executes actions using Moveit package
2. goal_recogniser - Code and infrastructure for training a goal recogniser network
3. inference - Code for running inference using trained models
4. inverse_model - Code for training the joint model 
5. kinect_ros - launch files for launching kinect cameras.
