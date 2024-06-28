import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from typing import List, Dict, Optional, Tuple, Callable
from efficientnet_pytorch import EfficientNet
from vint_train.models.vint.self_attention import PositionalEncoding

class NoMaD_ViNT(nn.Module):
    def __init__(
        self,
        context_size: int = 5,
        obs_encoder: Optional[str] = "efficientnet-b0",
        obs_encoding_size: Optional[int] = 512,
        mha_num_attention_heads: Optional[int] = 2,
        mha_num_attention_layers: Optional[int] = 2,
        mha_ff_dim_factor: Optional[int] = 4,
    ) -> None:
        """
        NoMaD ViNT Encoder class
        """
        super().__init__()
        self.obs_encoding_size = obs_encoding_size
        self.goal_encoding_size = obs_encoding_size
        self.context_size = context_size

        ##Initialize the observation encoder
        ##所以这里我理解efficientnet是直接安装efficientnet_pytorch后，通用encoder模型，这个模型并不会重新训练或者更新参数
        #!首先，EfficientNet 是一系列通过自动化机器学习技术优化的模型，常用于各种图像处理任务，并且通常包含大量预训练的参数。加载这样的预训练模型可以大幅度缩短训练时间，
        #! 提高模型在特定任务上的性能。但这并不意味着使用预训练模型就不能对其进行进一步的修改或优化，尤其是在模型需要被适应到具体应用场景中时。
        #!使用 replace_bn_with_gn 函数将 EfficientNet 中的所有 BatchNorm 替换为 GroupNorm，这主要是为了增强模型在不同训练条件（尤其是小批次训练条件）下的稳健性和泛化能力。这种替换虽然改变了模型的一部分结构，但不需要重新训练模型中的所有参数。
        if obs_encoder.split("-")[0] == "efficientnet":
            self.obs_encoder = EfficientNet.from_name(obs_encoder, in_channels=3) # context
            self.obs_encoder = replace_bn_with_gn(self.obs_encoder)
            self.num_obs_features = self.obs_encoder._fc.in_features
            self.obs_encoder_type = "efficientnet"
        else:
            raise NotImplementedError
        
        #!Initialize the goal-encoder。goal-encode 是当前obs和goal在维度上进行堆叠，所有是6通道
        self.goal_encoder = EfficientNet.from_name("efficientnet-b0", in_channels=6) # obs+goal
        self.goal_encoder = replace_bn_with_gn(self.goal_encoder)
        self.num_goal_features = self.goal_encoder._fc.in_features

        # Initialize compression layers if necessary
        #!按照论文描述，EfficientNet的输出应该是1000，self.num_obs_features= 1000， 而self.obs_encoding_size = 256
        #*这是通过条件性地添加一个全连接层（nn.Linear）来实现的，
        #*如果输入和输出特征尺寸不匹配，则这一层会将特征尺寸从 num_obs_features 或 num_goal_features 压缩至 obs_encoding_size 或 goal_encoding_size
        if self.num_obs_features != self.obs_encoding_size:
            self.compress_obs_enc = nn.Linear(self.num_obs_features, self.obs_encoding_size)
        else:
            self.compress_obs_enc = nn.Identity()
        
        if self.num_goal_features != self.goal_encoding_size:
            self.compress_goal_enc = nn.Linear(self.num_goal_features, self.goal_encoding_size)
        else:
            self.compress_goal_enc = nn.Identity()

        # Initialize positional encoding and self-attention layers
        self.positional_encoding = PositionalEncoding(self.obs_encoding_size, max_seq_len=self.context_size + 2)
        #!d_model：输入和输出的特征维度大小。在这里，它被设置为 self.obs_encoding_size，即观测数据的编码大小。d_model = 256
        #!nhead：注意力机制中的头数   mha_num_attention_heads = 4
        #! 全连接层的隐藏层大小。这里被设置为 mha_ff_dim_factor*self.obs_encoding_size    mha_ff_dim_factor = 4
        #!activation：激活函数。在这里，使用 GELU（Gaussian Error Linear Unit）作为激活函数。
        #!batch_first：输入张量的形状是否为 [batch_size, seq_length, feature_size]。如果设置为 True，则输入形状为 [batch_size, seq_length, feature_size]；如果设置为 False，则输入形状为 [seq_length, batch_size, feature_size]。在这里，设置为 True，表示输入的第一个维度是批次大小。
        #!norm_first：是否将 LayerNormalization 放在每个子层的输入之前。这是一个自定义参数，可能是您的实现中添加的额外逻辑。
        #*在 Transformer 模型中，每个编码器或解码器层通常包含两个子层,自注意力层（Self-Attention Layer）：用于在输入序列中建立全局依赖关系，以便每个位置可以关注到其他所有位置的信息。
        #*前馈神经网络（Feedforward Neural Network）：对每个位置的表示进行非线性变换和映射，增加模型的表达能力。
        #*self.sa_layer 是一个完整的 Transformer 编码器层，包含了一个自注意力层和一个前馈神经网络层。当您堆叠多个这样的层时，每个层都会有一个自注意力层和一个前馈神经网络层，但这些层不是相互独立的，而是通过残差连接和层归一化连接在一起的。
        self.sa_layer = nn.TransformerEncoderLayer(
            d_model=self.obs_encoding_size, 
            nhead=mha_num_attention_heads, 
            dim_feedforward=mha_ff_dim_factor*self.obs_encoding_size, 
            activation="gelu", 
            batch_first=True, 
            norm_first=True
        )
        #!self.sa_layer 定义了一个 Transformer 编码器的单个层，其中包含了自注意力机制和前馈神经网络。
        #!然后，nn.TransformerEncoder 使用多个这样的层堆叠成一个完整的编码器，其中 num_layers 指定了堆叠的层数。
        #!mha_num_attention_layers = 4  4layer
        self.sa_encoder = nn.TransformerEncoder(self.sa_layer, num_layers=mha_num_attention_layers)

        # Definition of the goal mask (convention: 0 = no mask, 1 = mask)
        self.goal_mask = torch.zeros((1, self.context_size + 2), dtype=torch.bool)
        self.goal_mask[:, -1] = True # Mask out the goal 
        self.no_mask = torch.zeros((1, self.context_size + 2), dtype=torch.bool) 
        self.all_masks = torch.cat([self.no_mask, self.goal_mask], dim=0)
        self.avg_pool_mask = torch.cat([1 - self.no_mask.float(), (1 - self.goal_mask.float()) * ((self.context_size + 2)/(self.context_size + 1))], dim=0)

    #!forward 函数是 nn.Module 类的一个标准方法（或标配方法），用于定义模型的前向传播逻辑。 每当你调用模型实例时，实际上会调用这个 forward 函数。
    def forward(self, obs_img: torch.tensor, goal_img: torch.tensor, input_goal_mask: torch.tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:

        device = obs_img.device

        # Initialize the goal encoding
        goal_encoding = torch.zeros((obs_img.size()[0], 1, self.goal_encoding_size)).to(device)
        
        # Get the input goal mask 
        if input_goal_mask is not None:
            goal_mask = input_goal_mask.to(device)

        # Get the goal encoding
        #!obs_img[:, 3*self.context_size:, :, :] 提取从第 9 个通道到最后一个通道的所有数据。  nomad中self.context_size的个数设置为3，上下文图片的个数，vint=5
        #!我理解obs_img 加上当前图片是4个，3+1， 那么通道的话是12个，3*self.context_size: 这个代表最后一个图片，也就是当前图片
        #!当前图片和目标图片堆叠后就是新的目标图片
        obsgoal_img = torch.cat([obs_img[:, 3*self.context_size:, :, :], goal_img], dim=1) # concatenate the obs image/context and goal image --> non image goal?
        obsgoal_encoding = self.goal_encoder.extract_features(obsgoal_img) # get encoding of this img 
        obsgoal_encoding = self.goal_encoder._avg_pooling(obsgoal_encoding) # avg pooling 
        
        if self.goal_encoder._global_params.include_top:
            obsgoal_encoding = obsgoal_encoding.flatten(start_dim=1)
            obsgoal_encoding = self.goal_encoder._dropout(obsgoal_encoding)
        obsgoal_encoding = self.compress_goal_enc(obsgoal_encoding)

        if len(obsgoal_encoding.shape) == 2:
            obsgoal_encoding = obsgoal_encoding.unsqueeze(1)
        assert obsgoal_encoding.shape[2] == self.goal_encoding_size
        goal_encoding = obsgoal_encoding
        
        # Get the observation encoding
        obs_img = torch.split(obs_img, 3, dim=1)
        obs_img = torch.concat(obs_img, dim=0)

        obs_encoding = self.obs_encoder.extract_features(obs_img)
        obs_encoding = self.obs_encoder._avg_pooling(obs_encoding)
        if self.obs_encoder._global_params.include_top:
            obs_encoding = obs_encoding.flatten(start_dim=1)
            obs_encoding = self.obs_encoder._dropout(obs_encoding)
        obs_encoding = self.compress_obs_enc(obs_encoding)
        obs_encoding = obs_encoding.unsqueeze(1)
        obs_encoding = obs_encoding.reshape((self.context_size+1, -1, self.obs_encoding_size))
        obs_encoding = torch.transpose(obs_encoding, 0, 1)
        obs_encoding = torch.cat((obs_encoding, goal_encoding), dim=1)
        
        # If a goal mask is provided, mask some of the goal tokens
        if goal_mask is not None:
            no_goal_mask = goal_mask.long()
            src_key_padding_mask = torch.index_select(self.all_masks.to(device), 0, no_goal_mask)
        else:
            src_key_padding_mask = None
        
        # Apply positional encoding 
        if self.positional_encoding:
            obs_encoding = self.positional_encoding(obs_encoding)

        obs_encoding_tokens = self.sa_encoder(obs_encoding, src_key_padding_mask=src_key_padding_mask)
        if src_key_padding_mask is not None:
            avg_mask = torch.index_select(self.avg_pool_mask.to(device), 0, no_goal_mask).unsqueeze(-1)
            obs_encoding_tokens = obs_encoding_tokens * avg_mask
        obs_encoding_tokens = torch.mean(obs_encoding_tokens, dim=1)

        return obs_encoding_tokens



#  Utils for Group Norm
## 在神经网络模型中，批量归一化（Batch Normalization, BatchNorm）是一种常用的技术，用于加速训练过程并改善模型的泛化能力。
# 批量归一化通过对小批量数据进行归一化处理，使得输出值的均值接近0，标准差接近1。然而，在某些情况下，批量归一化可能不是最佳选择，特别是在批量大小较小或者批量间的差异较大时。
# 这时，可以考虑使用组归一化（Group Normalization, GroupNorm）作为替代。
def replace_bn_with_gn(
    root_module: nn.Module,
    features_per_group: int=16) -> nn.Module:
    """
    Relace all BatchNorm layers with GroupNorm.
    """
    replace_submodules(
        root_module=root_module,
        predicate=lambda x: isinstance(x, nn.BatchNorm2d),
        func=lambda x: nn.GroupNorm(
            num_groups=x.num_features//features_per_group,
            num_channels=x.num_features)
    )
    return root_module


def replace_submodules(
        root_module: nn.Module,
        predicate: Callable[[nn.Module], bool],
        func: Callable[[nn.Module], nn.Module]) -> nn.Module:
    """
    Replace all submodules selected by the predicate with
    the output of func.

    predicate: Return true if the module is to be replaced.
    func: Return new module to use.
    """
    if predicate(root_module):
        return func(root_module)

    bn_list = [k.split('.') for k, m
        in root_module.named_modules(remove_duplicate=True)
        if predicate(m)]
    for *parent, k in bn_list:
        parent_module = root_module
        if len(parent) > 0:
            parent_module = root_module.get_submodule('.'.join(parent))
        if isinstance(parent_module, nn.Sequential):
            src_module = parent_module[int(k)]
        else:
            src_module = getattr(parent_module, k)
        tgt_module = func(src_module)
        if isinstance(parent_module, nn.Sequential):
            parent_module[int(k)] = tgt_module
        else:
            setattr(parent_module, k, tgt_module)
    # verify that all modules are replaced
    bn_list = [k.split('.') for k, m
        in root_module.named_modules(remove_duplicate=True)
        if predicate(m)]
    assert len(bn_list) == 0
    return root_module



    