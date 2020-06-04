# Useful-Policy-Invariant-Shaping-from-Arbitrary-Advice
This repository contains some of the codes from the paper [Useful Policy Invariant Shaping from Arbitrary Advice](https://ala2020.vub.ac.be/papers/ALA2020_paper_30.pdf).
## Requirements
The code is tested with python 3.7 and requires NumPy package.
## Run an experiment
You can run a specific experiment by running the experiment.py. In order to specify the experiment details such as the environment, the agent, learning parameters, etc., you can pass the name of your desired block from the experimental_config.ini to the config parser. We used a modified object-oriented version of RL-Glue for handling the agent and environemnt interactions[[1]](#1).
## References
<a id="1">[1]</a> 
Brian Tanner and Adam White. 
RL-Glue: Language-Independent Software for Reinforcement-Learning Experiments. 
Journal of Machine Learning Research, 10:2133--2136, 2009. [RL-Glue Page](https://sites.google.com/a/rl-community.org/rl-glue/Home?authuser=0)
