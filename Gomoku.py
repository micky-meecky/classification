import numpy as np
import random

# 定义棋盘大小
BOARD_SIZE = 15

# 定义棋盘
board = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=int)

def print_board():
    for i in range(BOARD_SIZE):
        for j in range(BOARD_SIZE):
            if board[i][j] == 0:
                print('.', end=' ')
            elif board[i][j] == 1:
                print('X', end=' ')
            elif board[i][j] == 2:
                print('O', end=' ')
        print()

def get_empty_positions():
    positions = []
    for i in range(BOARD_SIZE):
        for j in range(BOARD_SIZE):
            if board[i][j] == 0:
                positions.append((i, j))
    return positions

def ai_move():
    positions = get_empty_positions()
    return random.choice(positions)

# 玩家下棋
def player_move(x, y):
    if board[x][y] == 0:
        board[x][y] = 1
    else:
        print("Invalid move")

# AI下棋
def move():
    x, y = ai_move()
    if board[x][y] == 0:
        board[x][y] = 2
    else:
        print("Invalid move")

# 演示游戏
while True:
    # 模拟玩家下棋
    x, y = map(int, input("Enter your move: ").split())
    player_move(x, y)
    print_board()
    print()
    # AI 下棋
    move()
    print_board()
    print()
