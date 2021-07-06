"""
Tic Tac Toe Player
"""
import copy
import math

X = "X"
O = "O"
EMPTY = None


def initial_state():
    """
    Returns starting state of the board.
    """
    return [[EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY]]


def player(board):
    """
    Returns player who has the next turn on a board.
    """
    cells = board[0] + board[1] + board[2]
    if cells.count(O) < cells.count(X):
        return O
    else:
        return X


def actions(board):
    """
    Returns set of all possible actions (i, j) available on the board.
    """
    possible = set()
    for i in range(3):
        for j in range(3):
            if board[i][j] == EMPTY:
                possible.add((i, j))
    return possible


def result(board, action):
    """
    Returns the board that results from making move (i, j) on the board.
    """
    possible = actions(board)
    if action not in possible:
        raise
    new_board = copy.deepcopy(board)
    new_board[action[0]][action[1]] = player(board)
    return new_board


def winner(board):
    """
    Returns the winner of the game, if there is one.
    """
    # check rows, columns, diagonals
    for i in range(3):
        if set([board[i][0], board[i][1], board[i][2]]) in [set([X]), set([O])]:
            return board[i][0]
    for i in range(3):
        if set([board[0][i], board[1][i], board[2][i]]) in [set([X]), set([O])]:
            return board[0][i]
    if set([board[0][0], board[1][1], board[2][2]]) in [set([X]), set([O])]:
        return board[0][0]
    if set([board[0][2], board[1][1], board[2][0]]) in [set([X]), set([O])]:
        return board[0][2]
    return None


def terminal(board):
    """
    Returns True if game is over, False otherwise.
    """
    # if there's a winner or all the cells are full
    return bool(winner(board) or not actions(board))


def utility(board):
    """
    Returns 1 if X has won the game, -1 if O has won, 0 otherwise.
    """
    if winner(board) == X:
        return 1
    elif winner(board) == O:
        return -1
    else:
        return 0


def minimax(board):
    """
    Returns the optimal action for the current player on the board.
    """
    # use alpha-beta pruning with minimax
    if player(board) == X:
        return max_res(board, -float("inf"), float("inf"))[1]
    else:
        return mini_res(board, -float("inf"), float("inf"))[1]


def max_res(board, a, b):
    best = -float("inf")
    best_move = None
    if terminal(board):
        return utility(board), best_move

    for action in actions(board):
        new_best = mini_res(result(board, action), a, b)
        if new_best[0] > best:
            best = new_best[0]
            best_move = action
            if best >= b:
                return best, best_move
            a = max(best, a)
    return best, best_move


def mini_res(board, a, b):
    worst = float("inf")
    worst_move = None
    if terminal(board):
        return utility(board), worst_move

    for action in actions(board):
        new_worst = max_res(result(board, action), a, b)
        if new_worst[0] < worst:
            worst = new_worst[0]
            worst_move = action
            if worst <= a:
                return worst, worst_move
            b = min(worst, b)
    return worst, worst_move
