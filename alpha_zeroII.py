'''
只用 纯MCTS 作为AI棋手
'''

import pickle


from game import Board
from mcts_alphaZero import MCTSPlayer
from policy_value_net_numpy_pytorch import PolicyValueNetNumpy

from mcts_pure import MCTSPlayer as MCTS_Pure

		
# model_file="C:/APP/AlphaZero/exe/exe01/best_policy_8_8_5_new.model"
#
# try:
#     policy_param = pickle.load(open(model_file, 'rb'))
# except:
#     policy_param = pickle.load(open(model_file, 'rb'), encoding='bytes')  # To support python3
#
# #把模型参数类型转为numpy类型
# for k, v in policy_param.items():
#     policy_param[k] = v.numpy()  # v.cpu().numpy() 当模型是gup模型时候

def py_callback(board_state, currentPlayer, lastMove):
    """
    :param board_state: 当前棋盘状态
    :param currentPlayer: 当前玩家
    :param lastMove: 棋盘中最后一步落子位置
    :return: 当前玩家的下一步落子位置
    """
    states, sensible_moves = dealwithData(board_state)
    #计算下一步的落子位置
    move = run(states, sensible_moves, currentPlayer, lastMove)
    return move


def dealwithData(board_state):
    """
    :param board_state: 当前棋盘转态，例如"1212120000000000000000000000000000000000000000000000000000000000"
    :return:states(已经落子的{位置:玩家}棋盘状态),sensible_moves(没有落子的位置)
    """
    sensible_moves = []
    states = {}
    for i in range(len(board_state)):
        if not int(board_state[i]) == 0:
            states[i] = int(board_state[i])
        else:
            sensible_moves.append(i)
    return states, sensible_moves


def run(states, sensible_moves, currentPlayer, lastMove):
    n = 5
    width, height = 8, 8
    board = Board(width=width, height=height, n_in_row=n)
    board.init_board()

    board.states = states
    board.availables = sensible_moves
    board.current_player = currentPlayer
    board.last_move = lastMove


    #best_policy = PolicyValueNetNumpy(width, height, policy_param)
    #mcts_player = MCTSPlayer(best_policy.policy_value_fn, c_puct=5, n_playout=400)

    #只用纯MCTS
    mcts_player = MCTS_Pure(c_puct=5, n_playout=4000)   # n_playout参数 表示 搜索次数

    nextmove = mcts_player.get_action(board)

    return nextmove


if __name__ == '__main__':
    py_callback("12121202000000000000000000000000000000000000000000000000000000000", 1, 0)
