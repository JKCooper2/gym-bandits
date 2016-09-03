from gym.envs.registration import register
from gym.scoreboard.registration import add_task, add_group

from .bandit import BanditTenArmedRandomFixed
from .bandit import BanditTenArmedRandomRandom
from .bandit import BanditTenArmedRandomStochastic
from .bandit import BanditTwoArmedDeterministicFixed
from .bandit import BanditTwoArmedHighHighFixed
from .bandit import BanditTwoArmedHighLowFixed
from .bandit import BanditTwoArmedHighLowFixedNegative
from .bandit import BanditTwoArmedLowLowFixed

# Env registration
# ==========================
envs = [{'BanditTenArmedRandomFixed': ["10 armed bandit with random probabilities assigned to payouts"]},
        {'BanditTenArmedRandomRandom': ["10 armed bandit with random probabilities assigned to both payouts and rewards"]},
        {'BanditTenArmedRandomStochastic': ["10 armed bandit with random probabilities assigned to payouts, and reward are selected from a distribution"]},
        {'BanditTwoArmedDeterministicFixed': ["Simplest case where one bandit always pays, and the other always doesn't"]},
        {'BanditTwoArmedHighHighFixed': ["Stochastic version with a small difference between which bandit pays where both are good"]},
        {'BanditTwoArmedHighLowFixed': ["Stochastic version with a large difference between which bandit pays out of two choices"]},
        {'BanditTwoArmedHighLowFixedNegative': ["Stochastic version where one bandit pays out negative with a large percent"]},
        {'BanditTwoArmedLowLowFixed': ["Stochastic version with a small difference between which bandit pays where both are bad"]}]

for env in envs:
    env_name = env.keys()[0]

    register(
        id='{}-v0'.format(env_name),
        entry_point='gym_bandits:{}'.format(env_name),
        timestep_limit=1,
        nondeterministic=True,
    )

# Scoreboard registration
# ==========================
add_group(
    id='bandits',
    name='Bandits',
    description='Various N-Armed Bandit environments'
)

for env in envs:
    env_name = env.keys()[0]
    description = env.values()[0]

    add_task(
        id='{}-v0'.format(env_name),
        group='bandits',
        summary='{}'.format(description),
    )
