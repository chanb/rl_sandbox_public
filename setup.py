from setuptools import setup, find_packages

setup(name='rl_sandbox',
      version='1.0',
      packages=[package for package in find_packages()
                if package.startswith('rl_sandbox')],
      install_requires=[]
      )