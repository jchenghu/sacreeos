import torch

from sacreeos.reward_base import BaseRewardClass
from sacreeos.reward_base.base import Base


class GreedyBase(Base):
    """ Greedy based reward calculator. """

    DEFAULT_NSPI = 5

    """ Construct the greedy based reward calculator.
    
    :param nspi: int : number of samples per input (or image)
    """
    def __init__(self, nspi):
        super().__init__('greedy', 'nspi' + str(nspi), BaseRewardClass.GREEDY)
        self.nspi = nspi

    def input_check(self, pred_rewards, base_rewards):
        """ Check the correctness of the input shapes.

        :param pred_rewards: tensor : test reward scores with shape [batch_size, nsp, max_len]
        :param base_rewards: tensor : base reward scores with shape [batch_size, nsp, max_len]
        :return:
        """
        if (pred_rewards is None) or (base_rewards is None):
            raise TypeError("`base_rewards` and `pred_rewards` must be defined.")
        elif (not torch.is_tensor(pred_rewards)) or (not torch.is_tensor(base_rewards)):
            raise TypeError("`base_rewards` and `pred_rewards` must both be tensors.")
        elif (pred_rewards.size(1) != self.nspi) or (base_rewards.size(1) != self.nspi):
            raise ValueError("`pred_rewards` and `base_rewards` sizes are expected to be nspi: " + str(self.nspi)
                             + " in dim=1, got instead " + str(pred_rewards.size(1)) + ' and '
                             + str(base_rewards.size(1)) + " respectively.")

    def compute_based_reward(self, pred_rewards, base_rewards):
        """ Compute greedy based reward.

        :param pred_rewards: tensor : test reward scores with shape [batch_size, nsp]
        :param base_rewards: tensor : base reward scores with shape [batch_size, nsp]
        :return: tensor : results of the operation
        """
        # the operands order is not trivial, sacreeos expects logprobs and does not negate the final score
        return pred_rewards - base_rewards

    def get_nspi(self):
        """ Get the number of samples per input (or image)
        :return int :
        """
        return self.nspi
