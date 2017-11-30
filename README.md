# Bandit Environments

Series of n-armed bandit environments for the OpenAI Gym

Each env uses a different set of:
* Probability Distributions - A list of probabilities of the likelihood that a particular bandit will pay out
* Reward Distributions - A list of either rewards (if number) or means and standard deviations (if list) of the payout that bandit has

E.g. BanditTwoArmedHighLowFixed-v0 has `p_dist=[0.8, 0.2]`, `r_dist=[1, 1]`, meaning 80% of the time that action 0 is
selected it will payout 1, and 20% of the time action 2 is selected it will payout 1

You can access the distributions through the p_dist and r_dist variables using `env.p_dist` or `env.r_dist` if you want to match
your weights against the true values for plotting results of various algorithms



### Environments
* `BanditTwoArmedDeterministicFixed-v0`: Simplest case where one bandit always pays, and the other always doesn't
* `BanditTwoArmedHighLowFixed-v0`: Stochastic version with a large difference between which bandit pays out of two choices
* `BanditTwoArmedHighHighFixed-v0`: Stochastic version with a small difference between which bandit pays where both are good
* `BanditTwoArmedLowLowFixed-v0`: Stochastic version with a small difference between which bandit pays where both are bad
* `BanditTenArmedRandomFixed-v0`: 10 armed bandit with random probabilities assigned to payouts
* `BanditTenArmedRandomRandom-v0`: 10 armed bandit with random probabilities assigned to both payouts and rewards
* `BanditTenArmedUniformDistributedReward-v0`: 10 armed bandit with that always pays out with a reward selected from a uniform distribution
* `BanditTenArmedGaussian-v0`: 10 armed bandit mentioned on page 30 of [Reinforcement Learning: An Introduction](https://www.dropbox.com/s/b3psxv2r0ccmf80/book2015oct.pdf?dl=0) (Sutton and Barto)

### Installation
```bash
git clone git@github.com:JKCooper2/gym-bandits.git
cd gym-bandits
pip install .
```

To install using `requirements.txt` or `environment.yml` call:

```
git+https://github.com/JKCooper2/gym-bandits#egg=gym-bandits
```


In your gym environment
```python
import gym_bandits
env = gym.make("BanditTenArmedGaussian-v0") # Replace with relevant env
```