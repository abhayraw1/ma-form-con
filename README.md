# Formation Control Using MARL

## Some notes about this project

This project aims at training agents to cooperatively learn a policy to move to
a goal position while in a formation.

The rewards are sparse and two fold.
R_form: agents recieve this reward when the desired formation is achieved.
R_goal: agents recieve this reward when they have moved to the desired 
goal. This reward is only given when the agents have moved to the goal position
in the desired formation.

### The Environment

 - The environment will inherit the classic control gym envs.
 - The environment will be totally observable by the critic but partially
observable by the agents.
 - A function will convert the state of the environment to the observations of
the different agents.

It will have a goal, a random formation in the form of sides of the triangle

The main training algorithm will use DDPG + HER for training



## To do list:

## Setting up
