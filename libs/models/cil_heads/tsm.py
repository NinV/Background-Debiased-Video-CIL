import torch
import torch.nn as nn
from torch.nn import functional as F

from mmcv.runner import OPTIMIZER_BUILDERS, DefaultOptimizerConstructor
from mmcv.utils import SyncBatchNorm, _BatchNorm, _ConvNd
from mmaction.models.heads.tsm_head import TSMHead
from mmaction.models.builder import HEADS

from .inc_net import IncrementalNet, CosineIncrementalNet
from .cosine_linear import LSC
from ...losses import LSCLoss


inc_linear_layers = {
    'CosineLinear': CosineIncrementalNet,
    'SimpleLinear': IncrementalNet,
    'LocalSimilarityClassifier': LSC
}

@HEADS.register_module()
class IncrementalTSMHead(TSMHead):
    def __init__(self,
                 num_classes,
                 in_channels,
                 inc_head_config=dict(type='LocalSimilarityClassifier'),
                 num_segments=8,
                 loss_cls=dict(type='CrossEntropyLoss'),
                 spatial_type='avg',
                 consensus=dict(type='AvgConsensus', dim=1),
                 dropout_ratio=0.8,
                 init_std=0.001,
                 is_shift=True,
                 temporal_pool=False,
                 **kwargs):
        super(IncrementalTSMHead, self).__init__(
             num_classes,
             in_channels,
             num_segments=num_segments,
             loss_cls=loss_cls,
             spatial_type=spatial_type,
             consensus=consensus,
             dropout_ratio=dropout_ratio,
             init_std=init_std,
             is_shift=is_shift,
             temporal_pool=temporal_pool,
             **kwargs)

        self.inc_head_config = inc_head_config
        self.inc_head_config['in_features'] = in_channels

    def init_weights(self):
        # replace simple fc layer with incremental fc layer
        inc_head_config = self.inc_head_config.copy()
        head_type = inc_linear_layers[inc_head_config.pop('type')]
        self.fc_cls = head_type(**inc_head_config)
        self.fc_cls.update_fc(self.num_classes)

    def update_fc(self, nb_classes):
        if not hasattr(self.fc_cls, 'update_fc'):
            raise ValueError('Replace fc layer with incremental fc layer with "init_weights" method '
                             'before using "update_fc" method')

        self.fc_cls.update_fc(nb_classes)
        self.num_classes = nb_classes


@OPTIMIZER_BUILDERS.register_module()
class CILTSMOptimizerConstructor(DefaultOptimizerConstructor):
    """Modification of default Optimizer constructor in TSM model.
    support some custom layer such as: ConsineLinear and SimpleLinear

    This constructor builds optimizer in different ways from the default one.

    1. Parameters of the first conv layer have default lr and weight decay.
    2. Parameters of BN layers have default lr and zero weight decay.
    3. If the field "fc_lr5" in paramwise_cfg is set to True, the parameters
       of the last fc layer in cls_head have 5x lr multiplier and 10x weight
       decay multiplier.
    4. Weights of other layers have default lr and weight decay, and biases
       have a 2x lr multiplier and zero weight decay.
    """

    def add_params(self, params, model):
        """Add parameters and their corresponding lr and wd to the params.

        Args:
            params (list): The list to be modified, containing all parameter
                groups and their corresponding lr and wd configurations.
            model (nn.Module): The model to be trained with the optimizer.
        """
        # use fc_lr5 to determine whether to specify higher multi-factor
        # for fc layer weights and bias.
        fc_lr5 = self.paramwise_cfg['fc_lr5']
        first_conv_weight = []
        first_conv_bias = []
        normal_weight = []
        normal_bias = []
        lr5_weight = []
        lr10_bias = []
        bn = []

        conv_cnt = 0

        for m in model.modules():
            if isinstance(m, _ConvNd):
                m_params = list(m.parameters())
                conv_cnt += 1
                if conv_cnt == 1:
                    first_conv_weight.append(m_params[0])
                    if len(m_params) == 2:
                        first_conv_bias.append(m_params[1])
                else:
                    normal_weight.append(m_params[0])
                    if len(m_params) == 2:
                        normal_bias.append(m_params[1])
            elif isinstance(m, torch.nn.Linear):    # support SimpleLinear
                m_params = list(m.parameters())
                normal_weight.append(m_params[0])
                if len(m_params) == 2:
                    normal_bias.append(m_params[1])
            elif isinstance(m,
                            (_BatchNorm, SyncBatchNorm, torch.nn.GroupNorm)):
                for param in list(m.parameters()):
                    if param.requires_grad:
                        bn.append(param)

            elif isinstance(m, LSC):
                m_params = list(m.parameters())
                if fc_lr5:
                    lr5_weight.append(m_params[0])
                else:
                    normal_weight.append(m_params[0])
            elif isinstance(m, LSCLoss):
                eta = list(m.parameters())[0]
                if m.learnable_eta:
                    if fc_lr5:
                        lr5_weight.append(eta)
                    else:
                        normal_weight.append(eta)

            elif len(m._modules) == 0:
                if len(list(m.parameters())) > 0:
                    raise ValueError(f'New atomic module type: {type(m)}. '
                                     'Need to give it a learning policy')

        # pop the cls_head fc layer params
        # last_fc_weight = normal_weight.pop()
        # last_fc_bias = normal_bias.pop()
        # if fc_lr5:
        #     lr5_weight.append(last_fc_weight)
        #     lr10_bias.append(last_fc_bias)
        # else:
        #     normal_weight.append(last_fc_weight)
        #     normal_bias.append(last_fc_bias)

        params.append({
            'params': first_conv_weight,
            'lr': self.base_lr,
            'weight_decay': self.base_wd
        })
        params.append({
            'params': first_conv_bias,
            'lr': self.base_lr * 2,
            'weight_decay': 0
        })
        params.append({
            'params': normal_weight,
            'lr': self.base_lr,
            'weight_decay': self.base_wd
        })
        params.append({
            'params': normal_bias,
            'lr': self.base_lr * 2,
            'weight_decay': 0
        })
        params.append({'params': bn, 'lr': self.base_lr, 'weight_decay': 0})
        params.append({
            'params': lr5_weight,
            'lr': self.base_lr * 0.2,
            'weight_decay': self.base_wd
        })
        params.append({
            'params': lr10_bias,
            'lr': self.base_lr * 10,
            'weight_decay': 0
        })


@OPTIMIZER_BUILDERS.register_module()
class CILTSMOptimizerConstructorImprovised(DefaultOptimizerConstructor):
    """Modification of default Optimizer constructor in TSM model.
    Similar to CILTSMOptimizerConstructor

    This constructor builds optimizer in different ways from CILTSMOptimizerConstructor.
    1. Use "fc_lr_scale_factor" as learning rate multiplier for the classifier layer

        default setting for TSM model with dense layer as classifier: fc_lr_scale_factor=5.0
        When using LSC as classification head:
            Use high learning rate for classifier help model converge faster, very useful when using strong
            data augmentation
            In CBF step, lowering the classifier learning often results better NME accuracy while retain CNN accuracy
            high - low learning rate for classification head is a trade-off between previous task and more recent task
            Recommend:
                   fc_lr_scale_factor=5.0 when training with strong data augmentation, initial task and incremental
                   training step
                   fc_lr_scale_factor=[1.0, 0.2] for cbf steps


    """

    def add_params(self, params, model):
        """Add parameters and their corresponding lr and wd to the params.

        Args:
            params (list): The list to be modified, containing all parameter
                groups and their corresponding lr and wd configurations.
            model (nn.Module): The model to be trained with the optimizer.
        """
        # use fc_lr5 to determine whether to specify higher multi-factor
        # for fc layer weights and bias.
        fc_lr_scale_factor = self.paramwise_cfg['fc_lr_scale_factor']
        first_conv_weight = []
        first_conv_bias = []
        normal_weight = []
        normal_bias = []
        lr5_weight = []
        lr10_bias = []
        bn = []

        conv_cnt = 0

        for m in model.modules():
            if isinstance(m, _ConvNd):
                m_params = list(m.parameters())
                conv_cnt += 1
                if conv_cnt == 1:
                    first_conv_weight.append(m_params[0])
                    if len(m_params) == 2:
                        first_conv_bias.append(m_params[1])
                else:
                    normal_weight.append(m_params[0])
                    if len(m_params) == 2:
                        normal_bias.append(m_params[1])
            elif isinstance(m, torch.nn.Linear):    # support SimpleLinear
                m_params = list(m.parameters())
                normal_weight.append(m_params[0])
                if len(m_params) == 2:
                    normal_bias.append(m_params[1])
            elif isinstance(m,
                            (_BatchNorm, SyncBatchNorm, torch.nn.GroupNorm)):
                for param in list(m.parameters()):
                    if param.requires_grad:
                        bn.append(param)

            elif isinstance(m, LSC):
                m_params = list(m.parameters())
                lr5_weight.append(m_params[0])
            elif isinstance(m, LSCLoss):
                eta = list(m.parameters())[0]
                if m.learnable_eta:
                    lr5_weight.append(eta)

            elif isinstance(m, IncrementalNet):
                m_params = list(m.parameters())
                lr5_weight.append(m_params[0])
                lr10_bias.append(m_params[1])

            elif len(m._modules) == 0:
                if len(list(m.parameters())) > 0:
                    raise ValueError(f'New atomic module type: {type(m)}. '
                                     'Need to give it a learning policy')

        params.append({
            'params': first_conv_weight,
            'lr': self.base_lr,
            'weight_decay': self.base_wd
        })
        params.append({
            'params': first_conv_bias,
            'lr': self.base_lr * 2,
            'weight_decay': 0
        })
        params.append({
            'params': normal_weight,
            'lr': self.base_lr,
            'weight_decay': self.base_wd
        })
        params.append({
            'params': normal_bias,
            'lr': self.base_lr * 2,
            'weight_decay': 0
        })
        params.append({'params': bn, 'lr': self.base_lr, 'weight_decay': 0})
        params.append({
            'params': lr5_weight,
            'lr': self.base_lr * fc_lr_scale_factor,
            'weight_decay': self.base_wd
        })
