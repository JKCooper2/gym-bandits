from gym.envs.registration import register

from .bandit import BanditTenArmedRandomFixed
from .bandit import BanditTenArmedRandomRandom
from .bandit import BanditTenArmedGaussian
from .bandit import BanditTenArmedUniformDistributedReward
from .bandit import BanditTwoArmedDeterministicFixed
from .bandit import BanditTwoArmedHighHighFixed
from .bandit import BanditTwoArmedHighLowFixed
from .bandit import BanditTwoArmedLowLowFixed

environments = [['BanditTenArmedRandomFixed', 'v0'],
                ['BanditTenArmedRandomRandom', 'v0'],
                ['BanditTenArmedGaussian', 'v0'],
                ['BanditTenArmedUniformDistributedReward', 'v0'],
                ['BanditTwoArmedDeterministicFixed', 'v0'],
                ['BanditTwoArmedHighHighFixed', 'v0'],
                ['BanditTwoArmedHighLowFixed', 'v0'],
                ['BanditTwoArmedLowLowFixed', 'v0']]

for environment in environments:
    register(
        id='{}-{}'.format(environment[0], environment[1]),
        entry_point='gym_bandits:{}'.format(environment[0]),
    )
