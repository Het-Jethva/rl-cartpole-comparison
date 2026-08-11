# A Comparative Study of Value-Based and Policy-Based Reinforcement Learning on CartPole-v1

An empirical comparison of three reinforcement learning methods on `CartPole-v1`
(Gymnasium):

| Method | Representation |
| --- | --- |
| Tabular Q-learning | Discrete value table over a binned state space |
| DQN | Neural value function (replay buffer + target network) |
| REINFORCE with baseline | Direct policy, with a learned value-network baseline |

The three methods show three levels of approximation. The study measures sample
efficiency, final performance, wall-clock time, and stability across seeds.

## Status

Specification stage. See the project issues for the full spec.

## Deliverables

- Complete source code for the three methods and the experiment runner
- A research paper (IEEE two-column format)

## Reproduce

Instructions follow when the code lands.
