# Formation Control Using MARL

## Some notes about this project

This project aims at training agents to cooperatively learn a policy to move to
a goal position while in a formation.

The rewards are sparse and two fold.  
 - `R_form`: agents recieve this reward when the desired formation is achieved.
 - `R_goal`: agents recieve this reward when they have moved to the desired 
goal. This reward is only given when the agents have moved to the goal position
in the desired formation.

### The Environment

 - The environment will inherit the classic control gym envs.
 - The environment will be totally observable by the critic but partially
observable by the agents.
 - A function will convert the state of the environment to the observations of
the different agents.

#### State
The state of the environment will be denoted by a a vector indicating the
following:
This is the state of the system and will only be observed by the central critic.

1. Position vectors of the agents from the centroid of the formation(current)
2. Heading of the individual agents represented as the unit vector from 
the centroid of the formation(it will be the same as the world frame heading)

#### Agent Observations
This will be a dictionary of vectors representing the observed state of the
agents from their respective coordinate frames.

#### Goal
A triangle is sampled as the goal formation.
The goal is given as the relative position vector for agents from the 
centroid of the sampled goal formation.


It will have a goal, a random formation in the form of sides of the triangle.

The main training algorithm will use DDPG + HER for training



## To do list:
 - [ ] Make multiple agents
 - [x] Central critic
 - [ ] HER transitons
 - [ ] Rollouts
 - [ ] Training code
 - [ ] update code for maddpg
 - [ ] Exploration through critic

## Setting up
