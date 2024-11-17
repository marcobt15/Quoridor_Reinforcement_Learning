import heapq #in standard library
import copy #in standard library
import pandas as pd

#This class stores the information that will be used in the frontier
class State:
    def __init__(self, grid, heuristic, curr_location, num_keys = 0, path_cost = 0, path = []):
        self.path = path + [curr_location] #Stores the path from start to itself
        self.h = heuristic
        self.path_cost = path_cost
        self.grid = grid
        self.location = curr_location
        self.num_keys = num_keys

    def getPriorityValue(self):
        return self.path_cost + self.h
    
    #This is used for equality of two states
    def __eq__(self, other):
        return self.location == other.location and self.num_keys == other.num_keys
    
    #This is used for the order of the priority queue
    def __lt__(self, other):
        if self.getPriorityValue() == other.getPriorityValue():
            return self.h < other.h
        return self.getPriorityValue() < other.getPriorityValue()


#Retrieves the start and goal positions from the grid
def findStartAndGoal(grid):
    goals = []
    for row in range(len(grid)):
        for col in range(len(grid[row])):
            if grid[row][col] == "S":
                start = (row, col)
            elif grid[row][col] == "G":
                goals.append((row, col))

    return start, goals

#Returns the heuristic given the current position and goal
#The heuristic chosen is manhattan distance
def getHeuristic(curr_position, goals):
    shortest_m_dist = -1 #illegal move
    for goal in goals:
        m_dist_to_goal = abs(curr_position[0] - goal[0]) + abs(curr_position[1] - goal[1])
        if  m_dist_to_goal < shortest_m_dist and m_dist_to_goal > -1:
            shortest_m_dist = m_dist_to_goal
    return shortest_m_dist

#Checks if a point exists on the grid
def valid_point(point, row_length, col_length):
    if point[0] < 0 or point[1] < 0 or point[0] >= row_length or point[1] >= col_length:
        return False
    return True

#Gets the valid neighbours of a given location
def getNeighbours(grid, curr_location, has_key):
    directions = [(1,0), (0,1), (0,-1), (-1, 0)] #The four moveable directions

    neighbours = []

    for direction in directions:
        #adds the curr_location and direction tuples
        new_point = tuple(a + b for a, b in zip(direction, curr_location))

        #checks if the new point is valid
        #checks if the new point isn't a door, or if it a door then check if you have a key to open it
        if valid_point(new_point, len(grid), len(grid[0])) and (grid[new_point[0]][new_point[1]] != 'D' or has_key):
            #If all checks pass, then it is a valid neighbour
            neighbours.append(new_point)
    
    return neighbours


# The pathfinding function must implement A* search to find the goal state
def pathfinding(grid):
    start_grid = grid
    start_pos, goals_pos = findStartAndGoal(start_grid)

    # optimal_path is a list of coordinate of squares visited (in order)
    optimal_path = []
    # optimal_path_cost is the cost of the optimal path
    optimal_path_cost = 0
    # num_states_explored is the number of states explored during A* search
    num_states_explored = 0

    start_heuristic = getHeuristic(start_pos, goals_pos)
    start_state = State(start_grid, start_heuristic, start_pos)

    frontier = [start_state]
    explored = set() #for easy lookup

    while True:
        #There should be a solution
        if frontier == []:
            return [], -1, -1
        
        #use heapq to pop off best priority
        curr_state = heapq.heappop(frontier)
        num_states_explored += 1

        if curr_state.location in goals_pos:
            #curr_state stores the path cost and optimal path
            optimal_path_cost = curr_state.path_cost
            optimal_path = curr_state.path
            break

        # (x, y) + (num_keys, ) = (x, y, num_keys), easy way to concatenate tuples
        explored.add(curr_state.location + (curr_state.num_keys, ))

        curr_grid = curr_state.grid

        for node in getNeighbours(curr_state.grid, curr_state.location, curr_state.num_keys > 0):
            
            new_grid = copy.deepcopy(curr_grid)
            new_grid[node[0]][node[1]] = 'O'

            new_heuristic = getHeuristic(node, goals_pos)
            
            curr_keys = curr_state.num_keys
            if curr_grid[node[0]][node[1]] == 'K':
                curr_keys += 1
            elif curr_grid[node[0]][node[1]] == 'D':
                curr_keys -= 1
            
            new_path_cost = curr_state.path_cost + 1 #weight between two nodes will always be one
            
            new_state = State(new_grid, new_heuristic, node, curr_keys, new_path_cost, curr_state.path)

            #Checking for repeat states
            same_state = False
            for state in frontier:
                if new_state == state and new_path_cost < state.path_cost:
                    same_state = True
                    del new_state
                    new_state = copy.deepcopy(state)
                    #Since the path we're taking right now is better, update the path cost and path
                    new_state.path_cost = new_path_cost 
                    new_state.path = curr_state.path + [node]
                    break
            
            #If we've taken from the frontier or the new_state isn't in the frontier and isn't in explored
            if same_state or (new_state not in frontier and node + (curr_keys, ) not in explored):
                heapq.heappush(frontier, new_state)

    return optimal_path, optimal_path_cost, num_states_explored

def a_star(grid):
    optimal_path, optimal_path_cost, _ = pathfinding(grid)
    return optimal_path, optimal_path_cost