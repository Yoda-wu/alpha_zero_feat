# -*- coding: utf-8 -*-
"""
An implementation of the policyValueNet in PyTorch (tested in PyTorch 0.2.0 and 0.3.0)

@author: Junxiao Song
""" 

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

def set_learning_rate(optimizer, lr):
    """Sets the learning rate to the given value"""    
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

# 网络模型
class Net(nn.Module):
    """policy-value network module"""
    def __init__(self, board_width, board_height):
        super(Net, self).__init__()
        # 初始化棋盘宽高
        self.board_width = board_width
        self.board_height = board_height
        # 公共层
        self.conv1 = nn.Conv2d(4, 32, kernel_size=3, padding=1) # 32*3*3的卷积核，接受输入通道数为4，输出通道数为32.步长为1
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1) # 64*3*3的卷积核，接受输入通道数为32，输出通道数为64，步长为1
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        # 落棋策略层
        self.act_conv1 = nn.Conv2d(128, 4, kernel_size=1)
        self.act_fc1 = nn.Linear(4*board_width*board_height, board_width*board_height)
        # 状态价值层
        self.val_conv1 = nn.Conv2d(128, 2, kernel_size=1)
        self.val_fc1 = nn.Linear(2 * board_width * board_height, board_width*board_height)
        self.val_fc2 = nn.Linear(board_width*board_height, 1)

        #原先的
        # self.val_fc1 = nn.Linear(2*board_width*board_height, 64)
        # self.val_fc2 = nn.Linear(64, 1)
    
    def forward(self, state_input):
        # 公共层使用relu激活函数
        x = F.relu(self.conv1(state_input))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        # 落棋策略层卷积核使用relu激活，全连接层使用softmax
        x_act = F.relu(self.act_conv1(x))
        x_act = x_act.view(-1, 4*self.board_width*self.board_height)
        x_act = F.log_softmax(self.act_fc1(x_act),dim=1)  # x_act = F.log_softmax(self.act_fc1(x_act))  modified by haward 2018/03/12
        # 公共层使用relu激活函数，全连接层使用relu以及tanh
        x_val = F.relu(self.val_conv1(x))
        x_val = x_val.view(-1, 2*self.board_width*self.board_height)
        x_val = F.relu(self.val_fc1(x_val))
        x_val = F.tanh(self.val_fc2(x_val))
        return x_act, x_val


class PolicyValueNet():
    """policy-value network """
    def __init__(self, board_width, board_height, net_params=None, use_gpu=False):        #use_gpu=False
        self.use_gpu = use_gpu
        self.board_width = board_width
        self.board_height = board_height
        self.l2_const = 1e-4  # coef of l2 penalty 
        # the policy value net module
        if self.use_gpu:
            self.policy_value_net = Net(board_width, board_height).cuda()     # 这里用了上面的Net()
        else:
            self.policy_value_net = Net(board_width, board_height)
        self.optimizer = optim.Adam(self.policy_value_net.parameters(), weight_decay=self.l2_const)

        if net_params:
            self.policy_value_net.load_state_dict(net_params,strict=False)

    def policy_value(self, state_batch):
        """
        input: a batch of states
        output: a batch of action probabilities and state values 
        """
        if self.use_gpu:
            state_batch = Variable(torch.FloatTensor(state_batch).cuda())
            log_act_probs, value = self.policy_value_net(state_batch)
            act_probs = np.exp(log_act_probs.data.cpu().numpy())
            return act_probs, value.data.cpu().numpy()
        else:
            state_batch = Variable(torch.FloatTensor(state_batch))
            log_act_probs, value = self.policy_value_net(state_batch)
            act_probs = np.exp(log_act_probs.data.numpy())            
            return act_probs, value.data.numpy()
        

    def policy_value_fn(self, board):
        """
        input: board
        output: a list of (action, probability) tuples for each available action and the score of the board state
        """
        legal_positions = board.availables
        current_state = np.ascontiguousarray(board.current_state().reshape(-1, 4, self.board_width, self.board_height))
        if self.use_gpu:
            log_act_probs, value = self.policy_value_net(Variable(torch.from_numpy(current_state)).cuda().float())
            act_probs = np.exp(log_act_probs.data.cpu().numpy().flatten())
        else:
            log_act_probs, value = self.policy_value_net(Variable(torch.from_numpy(current_state)).float())
            act_probs = np.exp(log_act_probs.data.numpy().flatten())
        act_probs = zip(legal_positions, act_probs[legal_positions])
        value = value.data[0][0]
        return act_probs, value

    def train_step(self, state_batch, mcts_probs, winner_batch, lr):
        """训练步骤"""
        # 打包变量
        if self.use_gpu:
            state_batch = Variable(torch.FloatTensor(state_batch).cuda())
            mcts_probs = Variable(torch.FloatTensor(mcts_probs).cuda())
            winner_batch = Variable(torch.FloatTensor(winner_batch).cuda())
        else:
            state_batch = Variable(torch.FloatTensor(state_batch))
            mcts_probs = Variable(torch.FloatTensor(mcts_probs))
            winner_batch = Variable(torch.FloatTensor(winner_batch))

        # 初始化梯度为0
        self.optimizer.zero_grad()
        # 设置学习率
        set_learning_rate(self.optimizer, lr)

        # 前馈神经网络
        log_act_probs, value = self.policy_value_net(state_batch)
        # 定义损失函数 loss = (z - v)^2 - pi^T * log(p) + c||theta||^2 (Note: the L2 penalty is incorporated in optimizer)
        value_loss = F.mse_loss(value.view(-1), winner_batch)
        policy_loss = -torch.mean(torch.sum(mcts_probs*log_act_probs, 1))
        loss = value_loss + policy_loss
        # 反向传播和优化
        # 计算得到loss之后，反向传播，更新权重
        loss.backward()
        # 执行单个优化步骤，回传损失过程中会计算梯度，然后根据这些梯度更新参数
        self.optimizer.step()
       
        entropy = -torch.mean(torch.sum(torch.exp(log_act_probs) * log_act_probs, 1))
        return loss.data[0], entropy.data[0]

    def get_policy_param(self):
        net_params = self.policy_value_net.state_dict()
        return net_params
