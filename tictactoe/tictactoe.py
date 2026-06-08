"""
Tic Tac Toe Player
"""

X = "X"
O = "O"
EMPTY = None
memo = {}
winner_cache = {}


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
    # Count how many X's and O's are currently on the board
    x_count = sum(row.count(X) for row in board)
    o_count = sum(row.count(O) for row in board)

    return O if x_count > o_count else X


def actions(board):
    """
    Returns set of all possible actions (i, j) available on the board.
    """
    possible_moves = set()

    for i in range(3):
        for j in range(3):
            if board[i][j] is EMPTY:
                possible_moves.add((i, j))

    return possible_moves


def result(board, action):
    """
    Returns the board that results from making move (i, j) on the board.
    """

    i, j = action

    if i < 0 or i > 2 or j < 0 or j > 2 or board[i][j] is not EMPTY:
        raise ValueError("Invalid move")

    board_copy = [row[:] for row in board]
    board_copy[i][j] = player(board)

    return board_copy


def winner(board):
    """
    Returns the winner of the game, if there is one.
    """

    key = board_key(board)

    if key in winner_cache:
        return winner_cache[key]

    # Check rows and columns
    for i in range(3):
        if board[i][0] == board[i][1] == board[i][2] and board[i][0] is not EMPTY:
            result = board[i][0]
            winner_cache[key] = result
            return result

        if board[0][i] == board[1][i] == board[2][i] and board[0][i] is not EMPTY:
            result = board[0][i]
            winner_cache[key] = result
            return result

    # Check diagonals
    if board[0][0] == board[1][1] == board[2][2] and board[0][0] is not EMPTY:
        result = board[0][0]
        winner_cache[key] = result
        return result

    if board[0][2] == board[1][1] == board[2][0] and board[0][2] is not EMPTY:
        result = board[0][2]
        winner_cache[key] = result
        return result

    winner_cache[key] = None
    return None


def terminal(board):
    """
    Returns True if game is over, False otherwise.
    """
    return winner(board) is not None or all(
        cell is not EMPTY
        for row in board
        for cell in row
    )


def utility(board):
    """
    Returns 1 if X has won the game, -1 if O has won, 0 otherwise.
    """
    game_winner = winner(board)
    if game_winner == X:
        return 1
    elif game_winner == O:
        return -1
    else:
        return 0


def board_key(board):
    """
    Converts board into a hashable key for memoization.
    """
    return tuple(tuple(row) for row in board)


def minimax(board):
    """
    Returns the optimal action for the current player on the board.
    """
    memo.clear()
    if terminal(board):
        return None

    if board == initial_state():
        return (1, 1)

    current_player = player(board)
    best_move = None

    if current_player == X:
        max_val = float("-inf")
        alpha = float("-inf")
        beta = float("inf")

        for action in actions(board):

            move_val = min_value(
                result(board, action),
                alpha,
                beta
            )

            if move_val > max_val:
                max_val = move_val
                best_move = action

            alpha = max(alpha, max_val)
        return best_move

    else:
        min_val = float("inf")
        alpha = float("-inf")
        beta = float("inf")

        for action in actions(board):

            move_val = max_value(
                result(board, action),
                alpha,
                beta
            )

            if move_val < min_val:
                min_val = move_val
                best_move = action

            beta = min(beta, min_val)
        return best_move


def max_value(board, alpha, beta):

    key = board_key(board)
    if key in memo:
        return memo[key]

    if terminal(board):
        value = utility(board)
        memo[key] = value
        return value

    v = float("-inf")

    for action in actions(board):
        v = max(
            v,
            min_value(result(board, action), alpha, beta)
        )

        if v >= beta:
            memo[key] = v
            return v

        alpha = max(alpha, v)

    memo[key] = v
    return v


def min_value(board, alpha, beta):

    key = board_key(board)
    if key in memo:
        return memo[key]

    if terminal(board):
        value = utility(board)
        memo[key] = value
        return value

    v = float("inf")

    for action in actions(board):
        v = min(
            v,
            max_value(result(board, action), alpha, beta)
        )

        if v <= alpha:
            memo[key] = v
            return v

        beta = min(beta, v)

    memo[key] = v
    return v
