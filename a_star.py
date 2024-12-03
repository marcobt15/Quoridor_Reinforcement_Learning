import heapq
from collections import deque
import numpy as np

#agent can move up down left right
DIRECTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # up, down, left, right

class Node:
    def __init__(self, position, parent=None, g=0, h=0):
        self.position = position
        self.parent = parent
        self.g = g      #current cost
        self.h = h      #heuristic
        self.f = g + h  #totla cost

    def __lt__(self, other):
        return self.f < other.f

def findGoal(board_size, player):
    #player 1, goal is the 8th row
    if player == 'player_1':
        goal_row = 8
    else:
        #player two goal is 0th row
        goal_row = 0  

    #the entire row for a given agent is the 0 or 8th row
    goals = [(goal_row, col) for col in range(board_size)]  

    return goals

def a_star(start, current_agent, walls, board_size=9):
    walls = get_walls(board_size, walls)
    
    goals = findGoal(board_size, current_agent)
    
    #start node is current agent start and hueirstic is based off of all goal states
    start_node = Node(start, None, 0, heuristic(start, goals))
    
    #priority queue
    frontier = [start_node]
    explored = set()

    while frontier:
        #get lowest f value
        current_node = heapq.heappop(frontier)
        
        #goal state, rebuild path
        if current_node.position in goals:
            path, cost = reconstruct_path(current_node)
            return (path, cost) 

        explored.add(current_node.position)

        #check neighbours, loop though possible directions and check that square for neighbour
        for direction in DIRECTIONS:
            neighbour_pos = (current_node.position[0] + direction[0], current_node.position[1] + direction[1])

            #if out of bounds, skip neighbours
            if not is_valid_position(neighbour_pos, board_size):
                continue

            #do check for wall, if neighbour is wall, skip
            if is_wall_blocking(current_node.position, neighbour_pos, walls):
                continue

            #if already visited, skip
            if neighbour_pos in explored:
                continue

            #calculate g and h values, every move cost = 1
            g = current_node.g + 1  
            h = heuristic(neighbour_pos, goals)

            neighbor_node = Node(neighbour_pos, current_node, g, h)

            #add to frontier if not already there
            if not any(neighbor_node.position == n.position and neighbor_node.f >= n.f for n in frontier):
                heapq.heappush(frontier, neighbor_node)

    #no path found, important for checking valid moves
    return (-1, -1) 

def is_valid_position(position, board_size):
    x, y = position
    return 0 <= x < board_size and 0 <= y < board_size

def heuristic(position, goals):
    #min distance to any of the goal states
    return min(abs(position[0] - goal[0]) + abs(position[1] - goal[1]) for goal in goals)

def reconstruct_path(node):
    path = []
    cost = 0
    while node:
        cost += 1
        path.append(node.position)
        node = node.parent
        #reverse path and return cost
    return path[::-1], cost 

def is_wall_blocking(curr_pos, neighbor_pos, walls):
    x1, y1 = curr_pos
    x2, y2 = neighbor_pos

    #horizontal (left/right)
    if x1 == x2: 
        if y1 > y2:  #left
            return (x1, y2, '1') in walls
        else:  #right
            return (x1, y1, '1') in walls

    #vertical (up/down)
    elif y1 == y2:
        if x1 > x2:  #move up
            return (x2, y1, '0') in walls
        else:  #down
            return (x1, y1, '0') in walls

    return False


def get_walls(board_size, wall_positions):
    walls = set()
    for i in range(board_size - 1):
        for j in range(board_size - 1):
            if wall_positions[i][j][0] == 1:  #horizontal wall
                walls.add((i, j, '0'))  #normalpiece
                walls.add((i, j + 1, '0'))  #extended piece
            if wall_positions[i][j][1] == 1:  #vertical wall
                walls.add((i, j, '1'))  # Original wall
                walls.add((i + 1, j, '1'))  #extended piece
    return walls


def test_new_walls(player_positions, curr_wall_positions, wall_action_mask):
    player_1_pos = player_positions["player_1"]
    player_2_pos = player_positions["player_2"]

    horizontal_walls = curr_wall_positions[:, :, 0]
    vertical_walls = curr_wall_positions[:, :, 1]
    horizontal_flatten = horizontal_walls.flatten()
    vertical_flatten = vertical_walls.flatten()

    walls = np.concatenate([horizontal_flatten, vertical_flatten])

    for index, wall in enumerate(walls):
        #if there is already a wall there, don't test the position
        if wall == 1 or wall_action_mask[index] == 0:
            continue
    
        row, col, orientation = decode_wall_index(index)
        
        curr_wall_positions[row, col, orientation] = 1
        
        _, path_cost = a_star(player_1_pos, "player_1", curr_wall_positions)
        if path_cost == -1:
            wall_action_mask[index] = 0
            curr_wall_positions[row, col, orientation] = 0
            continue
        
        _, path_cost = a_star(player_2_pos, "player_2", curr_wall_positions)
        if path_cost == -1:
            wall_action_mask[index] = 0

        curr_wall_positions[row, col, orientation] = 0
        
def decode_wall_index(index):
    """Decodes a wall index into row, col, and orientation."""
    orientation = 0 if index < 64 else 1
    index = index % 64
    row = index // 8
    col = index % 8
    return row, col, orientation


def test_a_star():
    """Test the a_star function with a simple board setup."""
    board_size = 9

    wall_positions = np.zeros((8, 8, 2), dtype=int)
    wall_positions[0][3][1] = 1  
    wall_positions[1][3][0] = 1  
    wall_positions[0][4][1] = 1
    # Convert walls to the usable format
    walls = get_walls(board_size, wall_positions)

    # Player positions
    player_1_start = (0, 4)
    player_2_start = (8, 4)

    # Test A* for player 1
    print("Testing A* for Player 1...")
    path, cost = a_star(player_1_start, 'player_1', wall_positions, board_size)
    print(f"Player 1 Path: {path}, Cost: {cost}")

    # Test A* for player 2
    print("\nTesting A* for Player 2...")
    path, cost = a_star(player_2_start, 'player_2', wall_positions, board_size)
    print(f"Player 2 Path: {path}, Cost: {cost}")

    # Test movement checks
    print("\nTesting movement checks...")
    print(f"Player 1 trying to move right from, (false means move failed): (0,5), (0, 6)",
          not is_wall_blocking((0,5), (0, 6), walls))
    print(f"Player 1 trying to move down from, (false means move failed) (0,5), (1, 5):",
          not is_wall_blocking((0,5), (1, 5), walls))

# test_a_star()