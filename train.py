# -*- coding: utf-8 -*-
"""
An implementation of the training pipeline of AlphaZero for Gomoku

@author: Junxiao Song
""" 

from __future__ import print_function
import random
import numpy as np
import pickle  #import cPickle as pickle
from collections import defaultdict, deque
from game import Board, Game
#from policy_value_net import PolicyValueNet  # Theano and Lasagne
from policy_value_net_pytorch import PolicyValueNet  # Pytorch
from mcts_pure import MCTSPlayer as MCTS_Pure
from mcts_alphaZero import MCTSPlayer

import datetime
import logging

#save the log in the file "loss_win.log", filemode="w"表示 每次覆盖log文件
logging.basicConfig(filename="loss_win.log",level=logging.DEBUG,format="%(message)s",filemode="w")


class TrainPipeline():
    def __init__(self, init_model=None):
        # 棋盘和游戏参数
        self.board_width = 10  #6 #10
        self.board_height = 10 #6 #10
        self.n_in_row = 5     #4 #5
        self.board = Board(width=self.board_width, height=self.board_height, n_in_row=self.n_in_row)
        self.game = Game(self.board)
        # 训练参数
        self.learn_rate = 5e-3 # 学习率
        self.lr_multiplier = 1.0  # 学习率乘数，基于KL自适应调整学习率
        self.temp = 1.0 # the temperature param 温度参数
        self.n_playout = 800 # 每一步棋执行的MCTS模拟次数
        self.c_puct = 5
        self.buffer_size = 10000 # 队列中最大元素个数
        self.batch_size = 512 # mini-batch size for training
        self.data_buffer = deque(maxlen=self.buffer_size)       # 队列大小  
        self.play_batch_size = 1 
        self.epochs = 5 # 每个更新的训练步骤数
        self.kl_targ = 0.025 # KL目标
        self.check_freq = 50  #检查频率，每50次对局就对当前模型进行评估
        self.game_batch_num = 800 # 训练批量次数
        self.best_win_ratio = 0.0
        # 用于纯MCTS的模拟数，用来作为对手，来评估训练的策略网络。
        self.pure_mcts_playout_num = 1000  
        if init_model:
            # 增量训练
            # pickle.load(file)反序列化对象，将文件中的数据解析为一个Python对象
            policy_param = pickle.load(open(init_model, 'rb')) 
             # print("policy_param:",policy_param)
            selected_para = {
                'conv1.weight': policy_param['conv1.weight'],
                'conv1.bias': policy_param['conv1.bias'],
                'conv2.weight': policy_param['conv2.weight'],
                'conv2.bias': policy_param['conv2.bias'],
                'conv3.weight': policy_param['conv3.weight'],
                'conv3.bias': policy_param['conv3.bias'],
                'act_conv1.weight': policy_param['act_conv1.weight'],
                'act_conv1.bias': policy_param['act_conv1.bias'],
                'val_conv1.weight': policy_param['val_conv1.weight'],
                'val_conv1.bias': policy_param['val_conv1.bias'],
            }
            self.policy_value_net = PolicyValueNet(self.board_width, self.board_height, net_params = selected_para)
        else:
            # 从0开始训练
            self.policy_value_net = PolicyValueNet(self.board_width, self.board_height) 
        self.mcts_player = MCTSPlayer(self.policy_value_net.policy_value_fn, c_puct=self.c_puct, n_playout=self.n_playout, is_selfplay=1)

    def get_equi_data(self, play_data):
        """
        augment the data set by rotation and flipping
        play_data: [(state, mcts_prob, winner_z), ..., ...]"""
        extend_data = []
        for state, mcts_porb, winner in play_data:
            for i in [1,2,3,4]:
                # rotate counterclockwise 
                equi_state = np.array([np.rot90(s,i) for s in state])
                equi_mcts_prob = np.rot90(np.flipud(mcts_porb.reshape(self.board_height, self.board_width)), i)
                extend_data.append((equi_state, np.flipud(equi_mcts_prob).flatten(), winner))
                # flip horizontally
                equi_state = np.array([np.fliplr(s) for s in equi_state])
                equi_mcts_prob = np.fliplr(equi_mcts_prob)
                extend_data.append((equi_state, np.flipud(equi_mcts_prob).flatten(), winner))
        return extend_data
                
    def collect_selfplay_data(self, n_games=1):
        """collect self-play data for training"""
        for i in range(n_games):
            winner, play_data = self.game.start_self_play(self.mcts_player, temp=self.temp)
            play_data_zip2list = list(play_data)  # add by haward
            self.episode_len = len(play_data_zip2list)
            # augment the data
            play_data = self.get_equi_data(play_data_zip2list)
            self.data_buffer.extend(play_data)
                        
    def policy_update(self):
        """update the policy-value net"""
        mini_batch = random.sample(self.data_buffer, self.batch_size)
        state_batch = [data[0] for data in mini_batch]
        mcts_probs_batch = [data[1] for data in mini_batch]
        winner_batch = [data[2] for data in mini_batch]            
        old_probs, old_v = self.policy_value_net.policy_value(state_batch) 
        for i in range(self.epochs): 
            loss, entropy = self.policy_value_net.train_step(state_batch, mcts_probs_batch, winner_batch, self.learn_rate*self.lr_multiplier)
            new_probs, new_v = self.policy_value_net.policy_value(state_batch)
            kl = np.mean(np.sum(old_probs * (np.log(old_probs + 1e-10) - np.log(new_probs + 1e-10)), axis=1))  
            if kl > self.kl_targ * 4:   # early stopping if D_KL diverges badly
                break
        # adaptively adjust the learning rate
        if kl > self.kl_targ * 2 and self.lr_multiplier > 0.1:
            self.lr_multiplier /= 1.5
        elif kl < self.kl_targ / 2 and self.lr_multiplier < 10:
            self.lr_multiplier *= 1.5
            
        explained_var_old =  1 - np.var(np.array(winner_batch) - old_v.flatten())/np.var(np.array(winner_batch))
        explained_var_new = 1 - np.var(np.array(winner_batch) - new_v.flatten())/np.var(np.array(winner_batch))        
        print("kl:{:.5f},lr_multiplier:{:.3f},loss:{},entropy:{},explained_var_old:{:.3f},explained_var_new:{:.3f}".format(
                kl, self.lr_multiplier, loss, entropy, explained_var_old, explained_var_new))
        return loss, entropy
        
    def policy_evaluate(self, n_games=10,batch=0):
        """
        Evaluate the trained policy by playing games against the pure MCTS player
        Note: this is only for monitoring the progress of training
        """
        current_mcts_player = MCTSPlayer(self.policy_value_net.policy_value_fn, c_puct=self.c_puct, n_playout=self.n_playout)
        pure_mcts_player = MCTS_Pure(c_puct=5, n_playout=self.pure_mcts_playout_num)
        win_cnt = defaultdict(int)
        for i in range(n_games):
            winner = self.game.start_play(current_mcts_player, pure_mcts_player, start_player=i%2, is_shown=0)
            win_cnt[winner] += 1
        win_ratio = 1.0*(win_cnt[1] + 0.5*win_cnt[-1])/n_games
        print("batch_i:{}, num_playouts:{}, win: {}, lose: {}, tie:{}".format(batch, self.pure_mcts_playout_num, win_cnt[1], win_cnt[2], win_cnt[-1]))
        logging.debug("batch_i {} num_playouts {} win {} lose {} tie {}".format(batch, self.pure_mcts_playout_num, win_cnt[1], win_cnt[2], win_cnt[-1]))
        return win_ratio
    
    def run(self):
        """run the training pipeline"""
        try:
            for i in range(self.game_batch_num):                
                self.collect_selfplay_data(self.play_batch_size) # 自我对弈，训练一局就收集一局的数据
                print("batch_i:{}, episode_len:{}".format(i+1, self.episode_len))
                #logging.debug("batch_i:{}, episode_len:{}".format(i+1, self.episode_len))
                if len(self.data_buffer) > self.batch_size:
                    # 当队列数量大于mini—batch梯度下降法所需的最小批量数目时，我们随机从队列中挑选最小批量的数据进行更新
                    loss, entropy = self.policy_update()
                    logging.debug("batch_i {} loss {}".format(i+1, loss))
                # 检查当前模型的性能并且保存模型参数
                if (i+1) % self.check_freq == 0:
                    print("current self-play batch: {}".format(i+1))
                    win_ratio = self.policy_evaluate(batch= i+1)
                    net_params = self.policy_value_net.get_policy_param() # get model params
                    # 保存模型，序列化对象，并将结果数据流写入到文件对象中
                    pickle.dump(net_params, open('current_policy_10_10_5_new.model', 'wb'), pickle.HIGHEST_PROTOCOL) # save model param to file
                    if win_ratio > self.best_win_ratio:  # 胜率提高，更新参数
                        print("New best policy!!!!!!!!")
                        self.best_win_ratio = win_ratio
                        # 更新模型
                        pickle.dump(net_params, open('best_policy_10_10_5_new.model', 'wb'), pickle.HIGHEST_PROTOCOL) # update the best_policy
                        if self.best_win_ratio == 1.0 and self.pure_mcts_playout_num < 5000:
                            # 胜率达到100%之后纯MCTS模拟次数增加到1000，胜率归0
                            self.pure_mcts_playout_num += 1000
                            self.best_win_ratio = 0.0
        except KeyboardInterrupt:
            print('\n\rquit')
    

if __name__ == '__main__':
    # training_pipeline = TrainPipeline()   # 从零训练 二选一
    training_pipeline = TrainPipeline("best_policy_8_8_5_new.model")   # 增量训练 二选一
    time_start = datetime.datetime.now()
    print("开始时间：" + time_start.strftime('%Y.%m.%d-%H:%M:%S'))
    training_pipeline.run()
    time_end = datetime.datetime.now()
    print("结束时间：" + time_end.strftime('%Y.%m.%d-%H:%M:%S'))
    