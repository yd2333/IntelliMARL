# Introduce
Machine ToM was
proposed in the last decade, but few studies implemented it to running agents. To specify and implement
ToM, ToM nets were designed to predict other agents’ observation or future actions. The implementation of
ToM boosted the learning and performance in multi-agent task.

This project implemented the ToM Net in a decentralized way with Q-learning.
Our goal is to improve agents’ performance with ToM networks in a cooperative task.

# Environment
11 by 11 grid world

# Model
![Alt text](images/ToM.png?raw=true "ToM structure")
ToM we proposed is a network build on top of the original Q-Learning algorithm. ToM learn the prediction of trajectories of other agents and pass it to the policy net as part of the states.

# Result
![Alt text](images/loss.png?raw=true "Loss")
![Alt text](images/rewards.png?raw=true "Rewards")