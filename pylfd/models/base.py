
from abc import ABCMeta
from abc import abstractmethod


class MDP(object):
    """ MDP Model """
    def __init__(self, domain, ):
        super(MDP, self).__init__()
        self.domain = domain


class Reward(object):
    """ Reward function model """
    def __init__(self, arg):
        super(Reward, self).__init__()
        self.arg = arg

