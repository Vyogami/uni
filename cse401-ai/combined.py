#!/usr/bin/env python
# coding: utf-8

# # EXPERIMENT 1
# ### 1.1 SINGLE PLAYER GAME

# In[ ]:





# In[ ]:


import random as rand
def game():
  hidden_num = rand.randrange(0,100,1)
  print("random number generated")
  num_guess = 7
  while num_guess != 0:
    guess = int(input("Enter a number between 1 and 100: "))
    if guess == hidden_num:
      print("guessed right")
      break
    elif guess > hidden_num:
      print("wrong guess lower")
      num_guess -= 1
    else:
      print("wrong guess higher")
      num_guess -=1
  print("game over, the hidden number is ", hidden_num)



# In[ ]:


game()


# # Experiment 2

# 
# ### 2.1 WATER JUG PROBLEM

# ## Explaining Rules:
# ---
# * `jug2_capacity - y`  =>  means how much empty space does jug2 have
# *  `min(x, jug2_capacity - y)` => if x is 0 then we dont do any transfer, if
#   * if so we try to find what the amount of water we can transfer in jug2 without overflowing it and thus we substract that amount from jug1
# *` min(y, jug1_capacity - x)` => same for transfering water from jug 2 to 1
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 

# In[ ]:


from collections import deque

def calc_water_jug(initial_state, final_state, jug1_capacity, jug2_capacity):
    visited = set()
    visited.add(initial_state)
    queue = deque([(initial_state, [])])  # Queue to hold the states of the jugs along with the path
    result_path = []  # Stores the path leading to the final state

    while queue:
        current_state, path = queue.popleft()
        if current_state == final_state:
            result_path = path
            break
        x, y = current_state
        combinations = [
            (x, 0),  # Empty jug B
            (0, y),  # Empty jug A
            (jug1_capacity, y),  # Fill jug A
            (x, jug2_capacity),  # Fill jug B
            (x - min(x, jug2_capacity - y), y + min(x, jug2_capacity - y)),  # Pour from A to B
            (x + min(y, jug1_capacity - x), y - min(y, jug1_capacity - x))  # Pour from B to A
        ]
        rules = [
            "Empty jug B",
            "Empty jug A",
            "Fill jug A",
            "Fill jug B",
            "Pour from A to B",
            "Pour from B to A"
        ]
        for i, next_state in enumerate(combinations):
            if next_state not in visited:
                visited.add(next_state)
                next_path = path + [(i, next_state)]  # Record the current path along with the rule
                queue.append((next_state, next_path))  # Add the next state to the queue along with the updated path

    for step, (rule, state) in enumerate(result_path):
        print(f"Step {step + 1}: {rules[rule]} -> {state}")
    print(f"Total steps: {len(result_path)}")


# In[ ]:


capacityA = 4
capacityB = 3
start_state = (0, 0)  # Initial state
final_state = (2, 0)  # Target state
calc_water_jug(start_state, final_state, capacityA, capacityB)


# 
# ### 2.2 Maze problem using DFS (STACK)

# In[ ]:


from collections import deque
def solve_maze(maze, start, destination):
    paths = []
    stack = deque([(start, [start])])
    visited = set()

    while stack:
        (x, y), path = stack.pop()
        if (x, y) == destination:
          paths.append(path)
          continue
        if (x, y) in visited:
            continue
        visited.add((x, y))

        for dx, dy in [(1, 0), (0, 1), (-1, 0), (0, -1)]: #rules (D,R,U,L)
            new_x, new_y = x + dx, y + dy
            if 0 <= new_x < 5 and 0 <= new_y < 5 and maze[new_x][new_y] == 0: # checks validity
                stack.append(((new_x, new_y), path + [(new_x, new_y)]))

    if not paths:
      return "no paths found"
    else:
      return paths


# In[ ]:


maze = [[ 0, -1, -1, -1, -1],
        [ 0,  0, -1, -1, -1],
        [-1,  0,  0,  0,  0],
        [-1, -1,  0, -1,  0],
        [-1, -1,  0,  0,  0]]
start = (0,0)
destination = (4,4)
solve_maze(maze,start,destination)


# # EXPERIMENT 3
# ### 3.1 EIGHT PUZZLE PROBLEM (USING BEST FIRST SEARCH)

# In[ ]:


import heapq
from copy import deepcopy

def manhattan_distance(initial_state, goal_state):
    distance = 0
    for row in range(3):
        for col in range(3):
            if initial_state[row][col] != 0:
                target_row, target_col = divmod(initial_state[row][col] - 1, 3)
                distance += abs(target_row - row) + abs(target_col - col)
    return distance

def get_possible_moves(state):
    moves = []
    for row in range(3):
        for col in range(3):
            if state[row][col] == 0:
                zero_row, zero_col = row, col
                break

    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # Up, Down, Left, Right
    for dr, dc in directions:
        new_row, new_col = zero_row + dr, zero_col + dc
        if 0 <= new_row < 3 and 0 <= new_col < 3:
            new_state = deepcopy(state)
            new_state[zero_row][zero_col], new_state[new_row][new_col] = new_state[new_row][new_col], new_state[zero_row][zero_col]
            move = ["Up", "Down", "Left", "Right"][directions.index((dr, dc))]
            moves.append((tuple(map(tuple, new_state)), move))

    return moves

def solve_puzzle_best_first_search(initial_state, goal_state):
    open_list = [(manhattan_distance(initial_state, goal_state), initial_state, ["Start"])]
    heapq.heapify(open_list)
    visited = set()

    while open_list:
        _, current_state, path = heapq.heappop(open_list)
        current_state = tuple(map(tuple, current_state))

        if current_state == goal_state:
            return list(map(list, current_state)), path

        visited.add(current_state)

        for next_state, move in get_possible_moves(list(map(list, current_state))):
            next_path = deepcopy(path)
            next_path.append(move)
            next_state = tuple(map(tuple, next_state))

            if next_state not in visited:
                heapq.heappush(open_list, (manhattan_distance(next_state, goal_state), next_state, next_path))

    return None, None


# In[ ]:


initial_state = ((1, 2, 3),
                 (5, 6, 0),
                 (7, 8, 4))

goal_state = ((1, 2, 3),
              (4, 5, 6),
              (7, 8, 0))

final_state, path = solve_puzzle_best_first_search(initial_state, goal_state)

if final_state:
  print("Final state:")
  for row in final_state:
    print(" ".join(map(str, row)))
  print("Moves:", " -> ".join(path[1:]))
else:
  print("No solution found.")


# In[ ]:





# # Experiment 4 (Constraint Satisfaction)
# ## 4.1 Crypt Arithmetic

# In[ ]:


from itertools import permutations

def solve_cryptarithmetic(puzzle):

    words = puzzle.replace('+', ' ').replace('=', ' ').split()
    unique_letters = set(''.join(words))

    # Try all possible digit assignments
    for perm in permutations(range(10), len(unique_letters)):
        sol = dict(zip(unique_letters, perm))

        # Check if the assignment satisfies the puzzle
        if valid(sol, words):
            return [sol]

    return []

def valid(sol, words):
    # Ignore leading zeros
    values = [int(''.join(str(sol[c]) for c in word)) for word in words]
    return not any(val == 0 for val in values[:len(values) - 1]) and sum(values[:len(values) - 1]) == values[-1] # boolean
# Example usage
puzzle = "SEND + MORE == MONEY"
solutions = solve_cryptarithmetic(puzzle)

if solutions:
    print("Solutions:")
    for solution in solutions:
        print(solution)
else:
    print("No solution found.")


# ## 4.2 solving graph colouring problems

# In[ ]:


import networkx as nx
def graph_init(num_nodes, num_edges):
  graph = nx.Graph()
  nodes = range(1,num_nodes + 1)
  graph.add_nodes_from(nodes)
  for i in range(num_edges) :
    src = int(input("Enter the src node of edge " + str(i+1)+": "))
    dest = int(input("Enter the dest node of edge " + str(i+1)+": "))
    graph.add_edge(src,dest)
  print("\nnodes:",graph.nodes())
  print("\edges:",graph.edges())

  return graph

def solve_colour(graph):
  color_names = ["Red", "Blue", "Green", "Yellow", "Orange", "Purple", "Pink", "Brown", "Gray", "Cyan"]
  color_map ={}
  for node in graph.nodes():
      neighbor_colors = {color_map.get(neighbor) for neighbor in graph.neighbors(node)}
      available_colors = {color for color in range(len(color_names))} - neighbor_colors
      color_map[node] = min(available_colors)

  print("\nColor assignments:")
  for node, color in color_map.items():
      print(f"Node {node}: {color_names[color]}")

  num_colors = max(color_map.values()) + 1
  print(f"\nNumber of colors used: {num_colors}")



# main calling all the functions
num_nodes = int(input("enter the number of nodes in the graph: "))
num_edges = int (input("enter the number of edges in the graph: "))
graph = graph_init(num_nodes,num_edges)
solve_colour(graph)


# # Experiment 5 Game theroy
# ## 5.1 impliment min max algorithm
# There are several counters in a shared pile.
# Two players take alternating turns.
# On their turn, a player removes one, two, or three counters from the pile.
# The player that takes the last counter loses the game.

# In[ ]:





# # Experiment 6
# ## 6.1 fractional knapsack using greedy

# In[1]:


def knapsack_greedy(weights, values, capacity):

  # Create a list of items with their weight-to-value ratio.
  items = sorted(zip(weights, values), key=lambda x: x[1] / x[0], reverse=True)

  fractional_weights = [0] * len(weights)
  total_value = 0
  current_weight = 0

  for weight, value in items:
    if current_weight + weight <= capacity:
      fractional_weights[items.index((weight, value))] = 1
      current_weight += weight
      total_value += value
    else:
      fractional_weights[items.index((weight, value))] = (capacity - current_weight) / weight
      current_weight += (capacity - current_weight)
      total_value += value * (capacity - current_weight) / weight
      break

  return fractional_weights, total_value

# Example usage
weights = [2, 3, 1, 4]
values = [4, 6, 3, 5]
capacity = 5

fractional_weights, total_value = knapsack_greedy(weights, values, capacity)

print("Fractional weights of selected items:", fractional_weights)
print("Total value:", total_value)


# ## 6.2 knapsack using DP

# In[2]:


def knapsack_dp(weights, values, capacity):
  n = len(weights)
  dp = [[0 for _ in range(capacity + 1)] for _ in range(n + 1)]

  # Build the DP table
  for i in range(1, n + 1):
    for w in range(capacity + 1):
      if weights[i - 1] > w:
        dp[i][w] = dp[i - 1][w]  # Don't include the item if it exceeds weight limit
      else:
        dp[i][w] = max(dp[i - 1][w], values[i - 1] + dp[i - 1][w - weights[i - 1]])
        # Include the item only if it yields a higher value

  # Backtrack to find the selected items
  selected_weights = []
  w = capacity
  for i in range(n, 0, -1):
    if dp[i][w] != dp[i - 1][w]:
      selected_weights.append(weights[i - 1])
      w -= weights[i - 1]

  return dp[n][capacity], selected_weights

# Example usage
weights = [2, 3, 1, 4]
values = [4, 6, 3, 5]
capacity = 5

max_value, selected_weights = knapsack_dp(weights, values, capacity)

print("Maximum value:", max_value)
print("Selected weights:", selected_weights)


# # Experiment 7

# In[2]:


import math

# Representation of the board
EMPTY = 0
PLAYER_X = 1
PLAYER_O = 2

def minimax(board, depth, is_max_turn, alpha=-math.inf, beta=math.inf):
    # Check if the game has ended
    result = check_win(board)
    if result is not None:
        if result == PLAYER_X:
            return 1
        elif result == PLAYER_O:
            return -1
        else:
            return 0

    # If we've reached the maximum depth, evaluate the board
    if depth == 0:
        return evaluate_board(board)

    # Explore all possible moves
    if is_max_turn:
        max_eval = -math.inf
        for move in get_possible_moves(board, PLAYER_X):
            new_board = make_move(board, move, PLAYER_X)
            eval_score = minimax(new_board, depth - 1, False, alpha, beta)
            max_eval = max(max_eval, eval_score)
            alpha = max(alpha, eval_score)
            if beta <= alpha:
                break  # Pruning
        return max_eval
    else:
        min_eval = math.inf
        for move in get_possible_moves(board, PLAYER_O):
            new_board = make_move(board, move, PLAYER_O)
            eval_score = minimax(new_board, depth - 1, True, alpha, beta)
            min_eval = min(min_eval, eval_score)
            beta = min(beta, eval_score)
            if beta <= alpha:
                break  # Pruning
        return min_eval

def evaluate_board(board):
    # Simple evaluation function that counts the number of pieces for each player
    x_count = sum(row.count(PLAYER_X) for row in board)
    o_count = sum(row.count(PLAYER_O) for row in board)
    return x_count - o_count

def check_win(board):
    # Check rows
    for row in board:
        if row.count(PLAYER_X) == 3:
            return PLAYER_X
        elif row.count(PLAYER_O) == 3:
            return PLAYER_O

    # Check columns
    for col in range(3):
        if board[0][col] == board[1][col] == board[2][col] != EMPTY:
            return board[0][col]

    # Check diagonals
    if board[0][0] == board[1][1] == board[2][2] != EMPTY:
        return board[0][0]
    if board[0][2] == board[1][1] == board[2][0] != EMPTY:
        return board[0][2]

    # No winner yet
    return None

def get_possible_moves(board, player):
    moves = []
    for i in range(3):
        for j in range(3):
            if board[i][j] == EMPTY:
                moves.append((i, j))
    return moves

def make_move(board, move, player):
    new_board = [row[:] for row in board]
    new_board[move[0]][move[1]] = player
    return new_board

# Example usage
initial_board = [[EMPTY, EMPTY, EMPTY],
                 [EMPTY, EMPTY, EMPTY],
                 [EMPTY, EMPTY, EMPTY]]

max_depth = 9  # Maximum depth for exploration

# Player X (maximizing player) starts first
result = minimax(initial_board, max_depth, True)
print(f"The optimal move for Player X has an evaluation of: {result}")

# Player O (minimizing player) starts first
result = minimax(initial_board, max_depth, False)
print(f"The optimal move for Player O has an evaluation of: {result}")


# ## Tokenization , Stemming and Lemmatisation using NLTK after removing stop words

# In[12]:


import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer

# Download NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Your input sentence
sentence = input("")

# Tokenization
tokens = word_tokenize(sentence)
print("Tokens:", tokens)

# Removing stopwords
filtered_words = [word for word in tokens if word.lower() not in stopwords.words('english')]
print("After removing stopwords:", filtered_words)

# Stemming
stemmer = PorterStemmer()
stemmed_words = [stemmer.stem(word) for word in filtered_words]
print("Stemmed words:", stemmed_words)

# Lemmatization
lemmatizer = WordNetLemmatizer()
lemmatized_words = [lemmatizer.lemmatize(word) for word in filtered_words]
print("Lemmatized words:", lemmatized_words)


# # MIN MAX

# In[3]:


class Choice:
    def __init__(self, left, right):
        self.left = left
        self.right = right

class Terminal:
    def __init__(self, value):
        self.value = value

tree = Choice(
    Choice(Terminal(5), Terminal(10)),
    Choice(Terminal(-10), Terminal(-20))
)

def min_max(tree, max_player):
    if isinstance(tree, Choice):
        lv = min_max(tree.left, not max_player)
        rv = min_max(tree.right, not max_player)
        if max_player:
            return max(lv, rv)
        else:
            return min(lv, rv)
    else:
        return tree.value

print(min_max(tree, True))
print(min_max(tree, False))


# # TIC TAC TOE ALPHA BETA PRUNING

# In[4]:


MAX, MIN = 1000, -1000
def minimax(depth, nodeIndex, maximizingPlayer, values, alpha, beta):
    if depth == 3:
        return values[nodeIndex]
    if maximizingPlayer:
        best = MIN
        for i in range(0, 2):
            val = minimax(depth + 1, nodeIndex * 2 + i, False, values, alpha, beta)
            best = max(best, val)
            alpha = max(alpha, best)
            if beta <= alpha:
                break
        return best
    else:
        best = MAX
        for i in range(0, 2):
            val = minimax(depth + 1, nodeIndex * 2 + i, True, values, alpha, beta)
            best = min(best, val)
            beta = min(beta, best)
            if beta <= alpha:
                break
        return best

if __name__ == "__main__":
    values = [10, 15, 36, 9, 11, 2, 30, -1]
    print("The optimal value is :", minimax(0, 0, True, values, MIN, MAX))


# # TIC TAC TOE MIN MAX

# In[7]:


import math

class Board:
    def __init__(self):
        self.board = [[' ' for _ in range(3)] for _ in range(3)]
    
    def print_board(self):
        for row in self.board:
            print("|".join(row))
        print("-" * 5)
    
    def game_over(self):
        for row in self.board:
            if row.count('X') == 3 or row.count('O') == 3:
                return True
        for col in range(3):
            if self.board[0][col] == self.board[1][col] == self.board[2][col] and self.board[0][col] != ' ':
                return True
        if self.board[0][0] == self.board[1][1] == self.board[2][2] and self.board[0][0] != ' ':
            return True
        if self.board[0][2] == self.board[1][1] == self.board[2][0] and self.board[0][2] != ' ':
            return True
        return False
    
    def get_empty_cells(self):
        return [(i, j) for i in range(3) for j in range(3) if self.board[i][j] == ' ']
    
    def make_move(self, row, col, player):
        self.board[row][col] = player
    
    def undo_move(self, row, col):
        self.board[row][col] = ' '

def minimax(board, depth, is_maximizing_player, alpha, beta):
    if board.game_over() or depth == 0:
        if board.game_over():
            if is_maximizing_player:
                return -1
            else:
                return 1
        else:
            return 0
    if is_maximizing_player:
        max_eval = -math.inf
        for (row, col) in board.get_empty_cells():
            board.make_move(row, col, 'O')
            eval = minimax(board, depth - 1, False, alpha, beta)
            board.undo_move(row, col)
            max_eval = max(max_eval, eval)
            alpha = max(alpha, eval)
            if beta <= alpha:
                break
        return max_eval
    else:
        min_eval = math.inf
        for (row, col) in board.get_empty_cells():
            board.make_move(row, col, 'X')
            eval = minimax(board, depth - 1, True, alpha, beta)
            board.undo_move(row, col)
            min_eval = min(min_eval, eval)
            beta = min(beta, eval)
            if beta <= alpha:
                break
        return min_eval

def find_best_move(board):
    best_move = (-1, -1)
    best_eval = -math.inf
    for (row, col) in board.get_empty_cells():
        board.make_move(row, col, 'O')
        eval = minimax(board, 5, False, -math.inf, math.inf)  # Depth is set to 5
        board.undo_move(row, col)
        if eval > best_eval:
            best_eval = eval
            best_move = (row, col)
    return best_move

def play_tic_tac_toe():
    board = Board()
    while not board.game_over():
        board.print_board()
        row = int(input("Enter row: "))
        col = int(input("Enter column: "))
        board.make_move(row, col, 'X')
        if board.game_over():
            break
        best_move = find_best_move(board)
        board.make_move(best_move[0], best_move[1], 'O')
        board.print_board()
    if 'O' in board.board[0]:
        print("You lose!")
    elif 'X' in board.board[0]:
        print("You win!")
    else:
        print("It's a draw!")

play_tic_tac_toe()



# # STOP WORDS

# In[8]:


import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

def preprocess_text(text):
    tokens = word_tokenize(text)
    # Removal of stop words
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token.lower() not in stop_words]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    return tokens

text= "Tokenization is a crucial step in natural language processing. It involves breaking down text into words or smaller sub-texts known as tokens."
processed_text = preprocess_text(text)
print("Processed text:", processed_text)


# # BOW

# In[9]:


from nltk.tokenize import word_tokenize
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords

pd.set_option('max_colwidth', 100)

texts = ["Bag-of-words (BoW) is a statistical language model used to analyze text and documents based on word count."]

def preprocess(text):
    text = text.lower()
    words = word_tokenize(text)
    filtered_words = [word for word in words if word not in stopwords.words("english")]
    text = " ".join(filtered_words)
    return text

texts = [preprocess(text) for text in texts]
print(texts)

vectorizer = CountVectorizer()
bag_of_words = vectorizer.fit_transform(texts)
print(bag_of_words.toarray())


# # XOR GATE

# In[10]:


def xor_gate(a, b):
    return (a and not b) or (not a and b)

print("XOR Gate Truth Table:")
print("0 XOR 0 =", xor_gate(0, 0))
print("0 XOR 1 =", xor_gate(0, 1))
print("1 XOR 0 =", xor_gate(1, 0))
print("1 XOR 1 =", xor_gate(1, 1))


# In[ ]:




