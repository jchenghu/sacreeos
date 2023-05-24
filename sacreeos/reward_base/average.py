import torch

from sacreeos.reward_base import BaseRewardClass
from sacreeos.reward_base.base import Base


class AverageBase(Base):
    """ Average based reward calculator. """


    """ Construct the based reward calculator.
     
    :param nspi: int : number of samples per input (or image)
    """
    def __init__(self, nspi):
        super().__init__('average', 'nspi' + str(nspi), BaseRewardClass.AVERAGE)
        self.nspi = nspi

    def input_check(self, pred_rewards):
        """ Check the correctness of the input shapes.

        :param pred_rewards: tensor : test reward scores with shape [batch_size, nsp, max_len]
        :return: tensor :
        """
        if pred_rewards is None or not torch.is_tensor(pred_rewards):
            raise TypeError("`preds_reward` must be defined and be a Tensor.")
        elif pred_rewards.size(1) != self.nspi:
            raise ValueError("`preds_rewards` size expected of nspi: " + str(self.nspi) + " in dim=1, "
                             "got instead " + str(pred_rewards.size(1)))

    def compute_based_reward(self, pred_rewards):
        """ Compute average based reward.

        :param pred_rewards: tensor : test reward scores with shape [batch_size, nsp]
        :return: set : (result of the operation, base)
        """
        base = (pred_rewards.sum(dim=-1, keepdim=True) - pred_rewards) / (self.nspi - 1)
        return pred_rewards - base, base

    def get_nspi(self):
        """ Get the number of samples per input (or image)
        :return: int :
        """
        return self.nspi
