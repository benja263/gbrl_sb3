from typing import Any, Dict, List, Optional, Tuple, Type, Union

import catboost as cb
import gymnasium as gym
import numpy as np
import torch as th
import xgboost as xgb
from gymnasium import spaces
from stable_baselines3.common.distributions import (
    BernoulliDistribution, CategoricalDistribution, DiagGaussianDistribution,
    Distribution, MultiCategoricalDistribution,
    SquashedDiagGaussianDistribution, StateDependentNoiseDistribution,
    make_proba_distribution)
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.preprocessing import get_action_dim
from stable_baselines3.common.torch_layers import (BaseFeaturesExtractor,
                                                   FlattenExtractor)
from stable_baselines3.common.type_aliases import Schedule
from stable_baselines3.common.utils import is_vectorized_observation
from torch import nn

from gbrl.gbrl_wrapper import to_numpy

LOG_STD_MAX = 2
LOG_STD_MIN = -20

def grad_loss(predt: np.ndarray, dtrain: xgb.DMatrix):
    grads = dtrain.get_label()
    return -grads, np.ones_like(grads) 

class GradMultiObjective(cb.MultiTargetCustomObjective):
    def calc_ders_multi(self, approx, target, weight):
        dim = 1 
        if len(target.shape) > 1:
            dim = target.shape[1]
        return [(-grad) for grad in target], [-np.ones(dim) for grad in target]
        
class GradObjective(object):
    def calc_ders_range(self, approxes, targets, weights):
        assert len(approxes) == len(targets)
        if weights is not None:
            assert len(weights) == len(approxes)
        return [(-grad, -1) for grad in targets]
    
    
class GradMetric(object):
    def get_final_error(self, error, weight):
        return np.sqrt(error / (weight + 1e-38))

    def is_max_optimal(self):
        return False

    def evaluate(self, approxes, target, weight):
        assert len(approxes) == 1
        assert len(target) == len(approxes[0])
        approx = approxes[0]
        error_sum = 0.0
        weight_sum = 0.0
        for i in range(len(approx)):
            w = 1.0 if weight is None else weight[i]
            weight_sum += w
            error_sum += w * ((approx[i] - target[i])**2)
        return error_sum, weight_sum
    
class MultiRmseMetric(cb.MultiTargetCustomMetric):
    def get_final_error(self, error, weight):
        return np.sqrt(error / (weight + 1e-38))

    def is_max_optimal(self):
        return False

    def evaluate(self, approxes, target, weight):
        assert len(target) == len(approxes)
        assert len(target[0]) == len(approxes[0])

        error_sum = 0.0
        weight_sum = 0.0
        
        for i in range(len(approxes[0])):
            w = 1.0 if weight is None else weight[i]
            weight_sum += w
            for d in range(len(approxes)):
                error_sum += w * ((approxes[d][i] - target[d][i])**2)
        return error_sum, weight_sum
class ActorCriticPolicyBOOST(BasePolicy):
    """
    Policy class for actor-critic algorithms (has both policy and value prediction).
    Used by A2C, PPO and the likes.

    :param observation_space: Observation space
    :param action_space: Action space
    :param lr_schedule: Learning rate schedule (could be constant)
    :param net_arch: The specification of the policy and value networks.
    :param activation_fn: Activation function
    :param ortho_init: Whether to use or not orthogonal initialization
    :param use_sde: Whether to use State Dependent Exploration or not
    :param log_std_init: Initial value for the log standard deviation
    :param full_std: Whether to use (n_features x n_actions) parameters
        for the std instead of only (n_features,) when using gSDE
    :param sde_net_arch: Network architecture for extracting features
        when using gSDE. If None, the latent features from the policy will be used.
        Pass an empty list to use the states as features.
    :param use_expln: Use ``expln()`` function instead of ``exp()`` to ensure
        a positive standard deviation (cf paper). It allows to keep variance
        above zero and prevent it from growing too fast. In practice, ``exp()`` is usually enough.
    :param squash_output: Whether to squash the output using a tanh function,
        this allows to ensure boundaries when using gSDE.
    :param features_extractor_class: Features extractor to use.
    :param features_extractor_kwargs: Keyword arguments
        to pass to the features extractor.
    :param normalize_images: Whether to normalize images or not,
         dividing by 255.0 (True by default)
    :param optimizer_class: The optimizer to use,
        ``th.optim.Adam`` by default
    :param optimizer_kwargs: Additional keyword arguments,
        excluding the learning rate, to pass to the optimizer
    """

    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        lr_schedule: Schedule,
        tree_struct: Dict = None, 
        tree_optimizer: Dict = None,
        net_arch: Optional[List[Union[int, Dict[str, List[int]]]]] = None,
        activation_fn: Type[nn.Module] = nn.Tanh,
        ortho_init: bool = True,
        its_per_grad: int = 1,
        use_sde: bool = False,
        fw_type: str = 'cb',
        squash: bool = False,
        log_std_init: float = -2.0,
        shared_tree_struct: bool=True,
        full_std: bool = True,
        sde_net_arch: Optional[List[int]] = None,
        use_expln: bool = False,
        squash_output: bool = False,
        features_extractor_class: Type[BaseFeaturesExtractor] = FlattenExtractor,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        normalize_images: bool = True,
        optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        is_categorical: bool = False
    ):

        if optimizer_kwargs is None:
            optimizer_kwargs = {}
            # Small values to avoid NaN in Adam optimizer
            if optimizer_class == th.optim.Adam:
                optimizer_kwargs["eps"] = 1e-5

        super().__init__(
            observation_space,
            action_space,
            features_extractor_class,
            features_extractor_kwargs,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
            squash_output=squash_output,
        )

        # Default network architecture, from stable-baselines
        if net_arch is None:
            net_arch = [dict(vf=[64, 64])]

        self.net_arch = net_arch
        self.activation_fn = activation_fn
        self.ortho_init = ortho_init
        self.is_categorical = is_categorical
        self.iteration = 0

        self.its_per_grad = its_per_grad

        self.shared_tree_struct = shared_tree_struct

        self.logits_dim = get_action_dim(action_space)
        self.discrete = isinstance(action_space, gym.spaces.Discrete)
        if self.discrete:
            self.logits_dim = self.logits_dim*action_space.n
        self.lr = tree_optimizer['pg_optimizer']['pg_lr']
        self.vf_lr = tree_optimizer['value_optimizer']['vf_lr']
        self.max_depth = tree_struct['max_depth']
        self.features_extractor = features_extractor_class(self.observation_space, **self.features_extractor_kwargs)
        self.features_dim = self.features_extractor.features_dim

        self.normalize_images = normalize_images
        self.log_std_init = log_std_init
        self.bias = np.zeros((1, self.logits_dim + 1))

        self.fw_type = fw_type

        # Action distribution
        self.action_dist = SquashedDiagGaussianDistribution(get_action_dim(action_space)) if isinstance(action_space, spaces.Box) and squash else make_proba_distribution(action_space)  
        self._build(tree_struct, tree_optimizer, lr_schedule)
    
    def _build(self, tree_struct: Dict, tree_optimizer: Dict, lr_schedule: Schedule) -> None:
        """

        """ 
        self.model = [None] * (self.logits_dim + 1)
        if self.fw_type == 'cb':
            # for i in range(self.logits_dim + 1):
            self.model[0] = cb.CatBoostRegressor(iterations=self.its_per_grad,
                                                    depth=self.max_depth,
                                                    learning_rate=self.lr,
                                                    random_strength=0,
                                                    verbose=0,
                                                    # boosting_type='Ordered',
                                                    eval_metric=MultiRmseMetric(),
                                                    loss_function=GradMultiObjective())
            self.model[1] = cb.CatBoostRegressor(iterations=self.its_per_grad,
                                                    depth=self.max_depth,
                                                    learning_rate=self.vf_lr,
                                                    random_strength=0,
                                                    # boosting_type='Ordered',
                                                    verbose=0,
                                                    eval_metric=GradMetric(),
                                                    loss_function=GradObjective())
        if isinstance(self.action_dist, DiagGaussianDistribution) or isinstance(self.action_dist, SquashedDiagGaussianDistribution) :
            self.log_std = nn.Parameter(th.ones(self.action_dist.action_dim) * self.log_std_init, requires_grad=True)
            self.log_std_optimizer = th.optim.Adam([self.log_std], lr=lr_schedule(1))


    def forward(self, obs: Union[th.Tensor, np.ndarray], deterministic: bool = False) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        """
        Forward pass in all the networks (actor and critic)

        :param obs: Observation
        :param deterministic: Whether to sample or use deterministic actions
        :return: action, value and log probability of the action
        """
        # Preprocess the observation if needed
        distribution, values = self._get_action_dist_from_obs(obs)
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        return actions, values, log_prob

    def _get_action_dist_from_obs(self, obs: Union[th.Tensor, np.ndarray]) -> Distribution:
        """
        Retrieve action distribution given the latent codes.

        :param latent_pi: Latent code for the actor
        :return: Action distribution
        """
        if isinstance(obs, th.Tensor):
            obs = obs.detach().cpu().numpy()
        theta = np.repeat(self.bias, repeats=len(obs), axis=0)
        if self.iteration > 0:
            if self.fw_type == 'xgb':
                for i in range(self.logits_dim + 1):
                    theta[:, i] += self.model[i].predict(xgb.DMatrix(obs))
            else:
                theta[:, :-1] += self.model[0].predict(obs)
                theta[:, -1] += self.model[1].predict(obs)

        if len(np.shape(theta)) == 1:
            theta = theta[np.newaxis, :]
        mean_actions, values = th.tensor(theta[:, :-1]).float(), th.tensor(theta[:, -1]).float()

        if isinstance(self.action_dist, SquashedDiagGaussianDistribution):
            return self.action_dist.proba_distribution(mean_actions, self.log_std), values.float()
        elif isinstance(self.action_dist, DiagGaussianDistribution):
            return self.action_dist.proba_distribution(mean_actions, self.log_std), values.float()
        elif isinstance(self.action_dist, CategoricalDistribution):
            # Here mean_actions are the logits before the softmax
            # return self.action_dist.proba_distribution(action_logits=mean_actions)
            return self.action_dist.proba_distribution(action_logits=mean_actions), values.float()
        elif isinstance(self.action_dist, MultiCategoricalDistribution):
            # Here mean_actions are the flattened logits
            return self.action_dist.proba_distribution(action_logits=mean_actions)
        elif isinstance(self.action_dist, BernoulliDistribution):
            # Here mean_actions are the logits (before rounding to get the binary actions)
            return self.action_dist.proba_distribution(action_logits=mean_actions)
        elif isinstance(self.action_dist, StateDependentNoiseDistribution):
            raise NotImplementedError
            # return self.action_dist.proba_distribution(mean_actions, self.log_std, latent_pi)
            # return self.action_dist.proba_distribution(mean_actions, self.log_std, mean_actions)
        else:
            raise ValueError("Invalid action distribution")
        
    def get_total_iterations(self):
        return self.iteration * (self.logits_dim + 1)

    def evaluate_actions_train(self, obs: Union[th.Tensor, np.ndarray], actions: th.Tensor) -> Tuple[Union[th.Tensor, Tuple[th.Tensor, th.Tensor]], th.Tensor, th.Tensor, th.Tensor]:
        """
        Retrieve action distribution given the latent codes.

        :param latent_pi: Latent code for the actor
        :return: Action distribution
        """
        if isinstance(obs, th.Tensor):
            obs = obs.detach().cpu().numpy()
        theta = np.repeat(self.bias, repeats=len(obs), axis=0)
        if self.iteration > 0:
            if self.fw_type == 'xgb':
                for i in range(self.logits_dim + 1):
                    theta[:, i] += self.model[i].predict(xgb.DMatrix(obs))
            else:
                theta[:, :-1] += self.model[0].predict(obs)
                theta[:, -1] += self.model[1].predict(obs)
        mean_actions, values = th.tensor(theta[:, :-1]).float(), th.tensor(theta[:, -1]).float().squeeze()
        mean_actions = th.tensor(mean_actions, requires_grad=True)
        values = th.tensor(values, requires_grad=True)
        if isinstance(self.action_dist, SquashedDiagGaussianDistribution):
            distribution = self.action_dist.proba_distribution(mean_actions, self.log_std)
        elif isinstance(self.action_dist, DiagGaussianDistribution):
            distribution = self.action_dist.proba_distribution(mean_actions, self.log_std)
        elif isinstance(self.action_dist, CategoricalDistribution):
            distribution = self.action_dist.proba_distribution(action_logits=mean_actions)
        elif isinstance(self.action_dist, MultiCategoricalDistribution):
            # Here mean_actions are the flattened logits
            distribution =  self.action_dist.proba_distribution(action_logits=mean_actions)
        elif isinstance(self.action_dist, BernoulliDistribution):
            # Here mean_actions are the logits (before rounding to get the binary actions)
            distribution =  self.action_dist.proba_distribution(action_logits=mean_actions)
        elif isinstance(self.action_dist, StateDependentNoiseDistribution):
            raise NotImplementedError
        else:
            raise ValueError("Invalid action distribution")

        log_prob = distribution.log_prob(actions)
        return mean_actions, values, log_prob, distribution.entropy()

    def _predict(self, observation: Union[th.Tensor, np.ndarray], deterministic: bool = False) -> th.Tensor:
        """
        Get the action according to the policy for a given observation.

        :param observation:
        :param deterministic: Whether to use stochastic or deterministic actions
        :return: Taken action according to the policy
        """
        return self.get_distribution(observation).get_actions(deterministic=deterministic)


    def predict(
        self,
        observation: Union[np.ndarray, Dict[str, np.ndarray]],
        state: Optional[Tuple[np.ndarray, ...]] = None,
        episode_start: Optional[np.ndarray] = None,
        deterministic: bool = False,
    ) -> Tuple[np.ndarray, Optional[Tuple[np.ndarray, ...]]]:
        """
        Get the policy action from an observation (and optional hidden state).
        Includes sugar-coating to handle different observations (e.g. normalizing images).

        :param observation: the input observation
        :param state: The last hidden states (can be None, used in recurrent policies)
        :param episode_start: The last masks (can be None, used in recurrent policies)
            this correspond to beginning of episodes,
            where the hidden states of the RNN must be reset.
        :param deterministic: Whether or not to return deterministic actions.
        :return: the model's action and the next hidden state
            (used in recurrent policies)
        """
        # Switch to eval mode (this affects batch norm / dropout)
        self.set_training_mode(False)

        vectorized_env = None
        if not self.is_categorical:
            observation, vectorized_env = self.obs_to_tensor(observation)
        vectorized_env = is_vectorized_observation(observation, self.observation_space)

        with th.no_grad():
            actions = self._predict(observation, deterministic=deterministic)
        # Convert to numpy, and reshape to the original action shape
        actions = actions.cpu().numpy().reshape((-1, *self.action_space.shape))

        if isinstance(self.action_space, spaces.Box):
            if self.squash_output:
                # Rescale to proper domain when using squashing
                actions = self.unscale_action(actions)
            else:
                # Actions could be on arbitrary scale, so clip the actions to avoid
                # out of bound error (e.g. if sampling from a Gaussian distribution)
                actions = np.clip(actions, self.action_space.low, self.action_space.high)

        # Remove batch dimension if needed
        if not vectorized_env:
            actions = actions.squeeze(axis=0)

        return actions, state

    def evaluate_actions(self, obs: th.Tensor, actions: th.Tensor) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        """
        Evaluate actions according to the current policy,
        given the observations.

        :param obs:
        :param actions:
        :return: estimated value, log likelihood of taking those actions
            and entropy of the action distribution.
        """
        # Preprocess the observation if needed
        distribution, values = self._get_action_dist_from_obs(obs)
        log_prob = distribution.log_prob(actions)
        return values, log_prob, distribution.entropy()
    
    def get_num_trees(self):
        return self.iteration * self.its_per_grad

    def get_distribution(self, obs: Union[th.Tensor, np.ndarray]) -> Distribution:
        """
        Get the current policy distribution given the observations.

        :param obs:
        :return: the action distribution.
        """
        dist, _ = self._get_action_dist_from_obs(obs)
        return dist

    def predict_values(self, obs: Union[th.Tensor, np.ndarray]) -> th.Tensor:
        """
        Get the estimated values according to the current policy given the observations.

        :param obs:
        :return: the estimated values.
        """
        _, values, _ = self(obs)
        return values

    
    def get_schedule_learning_rates(self):
        return self.lr, self.lr
    
    def fit(self, obs: Union[th.Tensor, np.ndarray], policy_grads: np.ndarray, critic_grads: np.ndarray) -> None:
        obs, _ = to_numpy(obs)
        target_grads = np.concatenate([policy_grads, critic_grads[:, np.newaxis]], axis=-1)
        target_grads = target_grads.reshape((len(target_grads), self.logits_dim + 1))
        if self.fw_type == 'xgb':
            for i in range(self.logits_dim + 1):
                y = target_grads[:, i]
                params = {'verbose': False, 'eta': self.lr if i < self.logits_dim else self.vf_lr, 'max_depth': self.max_depth, 'device': 'cuda' }
                self.model[i] = xgb.train(params, xgb.DMatrix(obs, label=y), self.its_per_grad, obj=grad_loss, xgb_model=self.model[i])
        else:
            self.model[0].fit(obs, target_grads[:, :-1], init_model= self.model[0] if self.iteration > 0 else None)
            self.model[1].fit(obs, target_grads[:, -1], init_model= self.model[1] if self.iteration > 0 else None)
        self.iteration += 1

    def get_iteration(self):
        return self.iteration
    
    def fit_actor(self, obs: Union[th.Tensor, np.ndarray], policy_grads: np.ndarray) -> None:
        self.model.model.fit_policy(obs, policy_grads)

    def fit_critic(self, obs: Union[th.Tensor, np.ndarray], critic_grads: np.ndarray) -> None:
        self.model.model.fit_value_fnc(obs, critic_grads)
    
    def update_learning_rate(self, policy_learning_rate, value_learning_rate):
        self.model.adjust_learning_rates(policy_learning_rate, value_learning_rate)
    
