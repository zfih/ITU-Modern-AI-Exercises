# search.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util


class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return [s, s, w, s, w, w, s, w]


def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print "Start:", problem.getStartState()
    print "Is the start a goal?", problem.isGoalState(problem.getStartState())
    print "Start's successors:", problem.getSuccessors(problem.getStartState())
    """
    "*** YOUR CODE HERE ***"
    frontier = util.Stack()
    explored = []
    roads = {(-1, -1): []}

    frontier.push(problem.getStartState())

    while not frontier.isEmpty():
        current_state = frontier.pop()

        if problem.isGoalState(current_state):
            return roads[current_state]

        for state in problem.getSuccessors(current_state):
            if state[0] not in frontier.list and state[0] not in explored:

                if current_state in roads:
                    road = roads[current_state][:]
                else:
                    road = []

                frontier.push(state[0])
                road.append(state[1])
                roads[state[0]] = road[:]

        explored.append(current_state)


def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    frontier = util.Queue()
    explored = []
    roads = {(-1, -1): []}

    frontier.push(problem.getStartState())

    while not frontier.isEmpty():
        current_state = frontier.pop()

        if problem.isGoalState(current_state):
            return roads[current_state]

        for state in problem.getSuccessors(current_state):
            if state[0] not in frontier.list and state[0] not in explored:

                if current_state in roads:
                    road = roads[current_state][:]
                else:
                    road = []

                frontier.push(state[0])
                road.append(state[1])
                roads[state[0]] = road[:]

        explored.append(current_state)


def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()


def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0


def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    state = problem.getStartState()

    frontier = util.PriorityQueue()
    frontier.update(state, 0)

    came_from = {}
    cost_so_far = {}
    came_from[state] = (None, 'Stop')
    cost_so_far[state] = 0

    states = {}
    states[state] = (state, 'None', 0)

    while not frontier.isEmpty():
        current_state = frontier.pop()

        if problem.isGoalState(current_state):
            road = []
            prev_pos, act = came_from[current_state]
            while prev_pos is not None:
                print act
                road.append(act)
                prev_pos, act = came_from[prev_pos]

            road.reverse()
            return road

        for next_state in problem.getSuccessors(current_state):
            states[next_state[0]] = next_state
            new_cost = cost_so_far[current_state] + next_state[2]
            if next_state[0] not in cost_so_far or new_cost < cost_so_far[next_state[0]]:
                cost_so_far[next_state[0]] = new_cost
                priority = new_cost + heuristic(next_state[0], problem)
                frontier.update(next_state[0], priority)
                came_from[next_state[0]] = (current_state, next_state[1])


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
