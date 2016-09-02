from setuptools import setup, find_packages
import sys
import os

# Don't import gym module here, since deps may not be installed
for package in find_packages():
    if '_gym_' in package:
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), package))

from package_info import USERNAME, VERSION

setup(name='{}_{}'.format(USERNAME, 'gym_bandits'),
      version=VERSION,
      description='Gym User Env - Various N-Armed Bandit Problems',
      url='https://github.com/jkcooper2/gym_bandits',
      author='Jesse Cooper',
      author_email='jesse_cooper@hotmail.com',
      license='MIT License',
      packages=[package for package in find_packages() if package.startswith(USERNAME)],
      package_data={},
      zip_safe=False,
      install_requires=['gym>=0.2.3'],
)
