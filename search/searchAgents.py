# searchAgents.py
# ---------------
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
This file contains all of the agents that can be selected to control Pacman.  To
select an agent, use the '-p' option when running pacman.py.  Arguments can be
passed to your agent using '-a'.  For example, to load a SearchAgent that uses
depth first search (dfs), run the following command:

> python pacman.py -p SearchAgent -a fn=depthFirstSearch

Commands to invoke other search strategies can be found in the project
description.

Please only change the parts of the file you are asked to.  Look for the lines
that say

"*** YOUR CODE HERE ***"

The parts you fill in start about 3/4 of the way down.  Follow the project
description for details.

Good luck and happy searching!
"""

from game import Directions
from game import Agent
from game import Actions
from DQNagent import DQNagent
import copy
import numpy as np
import GA_util

import random
import util
import time
import search
import math


class GoWestAgent(Agent):
    "An agent that goes West until it can't."

    def getAction(self, state):
        "The agent receives a GameState (defined in pacman.py)."
        if Directions.WEST in state.getLegalPacmanActions():
            return Directions.WEST
        else:
            return Directions.STOP


class BTNode:

    def __init__(self, children):
        self.children = children


class BTSequence(BTNode):

    def evaluate(self):
        for node in self.children:
            result = node.evaluate()
            if not result:
                return False
        return result


class BTSelector(BTNode):

    def evaluate(self):
        for node in self.children:
            result = node.evaluate()
            if result:
                return result
        return False


class BTLeaf(BTNode):

    def __init__(self, function, params=None):
        self.function = function
        self.params = params

    def evaluate(self):
        if self.params is None:
            return self.function()
        return self.function(self.params)


class BTDecorator(BTNode):
    def __init__(self, child, function):
        self.children = child
        self.function = function

    def evaluate(self):
        return self.function(self.children)


def fixedGhostPos(ghost):
    pos = ghost.getPosition()
    dir = ghost.getDirection()

    if pos[0] % 1.0 != 0.5 and pos[1] % 1.0 != 0.5:
        return pos

    if dir == "North":
        pos = (pos[0], pos[1] - 0.5)
    elif dir == "South":
        pos = (pos[0], pos[1] + 0.5)
    elif dir == "West":
        pos = (pos[0] + 0.5, pos[1])
    else:
        pos = (pos[0] - 0.5, pos[1])

    return pos


def checkGhostsNearbyState(state):
    for act in state.getLegalActions():
        possibleStates = [state]
        next_state = state.generatePacmanSuccessor(act)

        for ghostAct in next_state.getLegalActions(1):
            for ghost2Act in next_state.getLegalActions(2):
                tmp_state1 = next_state.generateSuccessor(1, ghostAct)
                if tmp_state1.isLose():
                    return True

                tmp_state2 = tmp_state1.generateSuccessor(2, ghost2Act)
                if tmp_state2.isLose():
                    return True

                possibleStates.append(tmp_state2)

        for check_state in possibleStates:
            for inner_act in check_state.getLegalActions():
                next_next_state = check_state.generatePacmanSuccessor(inner_act)
                for ghostAct in next_next_state.getLegalActions(1):
                    for ghost2Act in next_next_state.getLegalActions(2):
                        tmp_state1 = next_next_state.generateSuccessor(1, ghostAct)
                        if tmp_state1.isLose():
                            return True

                        tmp_state2 = tmp_state1.generateSuccessor(2, ghost2Act)
                        if tmp_state2.isLose():
                            return True

    return False


class BTAgent(Agent):
    def getAction(self, state):
        def fleeFromNearbyGhosts():
            badDirections = ['Stop']
            ghostNearby = False

            roads = []

            for act in state.getLegalActions():
                possibleStates = [state]
                next_state = state.generatePacmanSuccessor(act)
                road = [act]

                for ghostAct in next_state.getLegalActions(1):
                    for ghost2Act in next_state.getLegalActions(2):
                        tmp_state1 = next_state.generateSuccessor(1, ghostAct)
                        if tmp_state1.isLose():
                            ghostNearby = True
                            badDirections.append(act)
                            continue

                        tmp_state2 = tmp_state1.generateSuccessor(2, ghost2Act)
                        if tmp_state2.isLose():
                            ghostNearby = True
                            badDirections.append(act)

                        possibleStates.append(tmp_state2)

                for check_state in possibleStates:
                    for inner_act in check_state.getLegalActions():
                        next_next_state = check_state.generatePacmanSuccessor(inner_act)
                        road.append(inner_act)
                        for ghostAct in next_next_state.getLegalActions(1):
                            for ghost2Act in next_next_state.getLegalActions(2):
                                tmp_state1 = next_next_state.generateSuccessor(1, ghostAct)
                                if tmp_state1.isLose():
                                    ghostNearby = True
                                    badDirections.append(act)
                                    continue

                                tmp_state2 = tmp_state1.generateSuccessor(2, ghost2Act)
                                if tmp_state2.isLose():
                                    ghostNearby = True
                                    badDirections.append(act)
                                    continue
                                roads.append(road)

            if ghostNearby:
                random.shuffle(roads)
                for road in roads:
                    if road[0] not in badDirections:
                        return road[0]

            return False

        def checkGhostsNearby():
            return checkGhostsNearbyState(state)

        def bfsToNearestPill():
            frontier = util.Queue()
            explored = []
            roads = {}

            states = {}
            states[state.getPacmanPosition()] = state

            frontier.push(state.getPacmanPosition())

            while not frontier.isEmpty():
                current_pos = frontier.pop()
                current_state = states[current_pos]
                explored.append(current_pos)

                if current_state.getFood()[current_pos[0]][current_pos[1]]:
                    return roads[current_pos][0]

                for act in current_state.getLegalActions():
                    next_state = current_state.generatePacmanSuccessor(act)
                    next_pos = next_state.getPacmanPosition()
                    states[next_pos] = next_state

                    if next_pos not in frontier.list and next_pos not in explored:
                        if current_pos in roads:
                            road = roads[current_pos][:]
                        else:
                            road = []

                        frontier.push(next_pos)
                        road.append(act)
                        roads[next_pos] = road[:]

                        if current_state.hasFood(next_pos[0], next_pos[1]):  # how does duck-typing fail this?
                            return roads[next_pos][0]

            return False

        def checkForScaredGhosts():
            ghosts = state.getGhostStates()
            killableGhost = False

            for ghost in ghosts:
                if ghost.scaredTimer is 0:
                    continue
                distance = util.manhattanDistance(ghost.getPosition(), state.getPacmanPosition())

                # if distance to ghost is less than the time steps to get to ghost then we can kill it
                if distance < ghost.scaredTimer:
                    killableGhost = True

            if killableGhost:
                return True
            return False

        def AStarToGoal(goal):
            # Modified version from search.py

            frontier = util.PriorityQueue()
            frontier.update(state.getPacmanPosition(), 0)

            came_from = {}
            cost_so_far = {}
            came_from[state.getPacmanPosition()] = (None, 'Stop')
            cost_so_far[state.getPacmanPosition()] = 0

            states = {}
            states[state.getPacmanPosition()] = state

            while not frontier.isEmpty():
                current_pos = frontier.pop()

                if current_pos == goal:
                    road = []
                    prev_pos, act = came_from[current_pos]
                    while prev_pos is not None:
                        road.append(act)
                        prev_pos, act = came_from[prev_pos]
                    return road.pop()  # pop the last element as road is currently reversed.

                current_state = states[current_pos]

                for act in current_state.getLegalActions():
                    next_state = current_state.generatePacmanSuccessor(act)
                    states[next_state.getPacmanPosition()] = next_state
                    new_cost = cost_so_far[current_state.getPacmanPosition()] + 1
                    if next_state.getPacmanPosition() not in cost_so_far or new_cost < cost_so_far[
                        next_state.getPacmanPosition()]:
                        cost_so_far[next_state.getPacmanPosition()] = new_cost
                        priority = new_cost + util.manhattanDistance(goal, next_state.getPacmanPosition())
                        frontier.update(next_state.getPacmanPosition(), priority)
                        came_from[next_state.getPacmanPosition()] = (current_pos, act)
            return False

        def aStarToNearestCapsule():
            if len(state.getCapsules()) == 0:
                return False

            goal_state = None

            distance = 1000000000
            for cap in state.getCapsules():
                manhattan = util.manhattanDistance(cap, state.getPacmanPosition())
                if manhattan < distance:
                    distance = manhattan
                    goal_state = cap

            if goal_state is None:
                goal_state = random.choice(state.getCapsules())

            return AStarToGoal(goal_state)

        def aStarToNearestScaredGhost():
            if len(state.getGhostStates()) == 0:
                return False

            spawn = [(11, 5), (10, 5), (10, 6), (9, 6), (9, 5), (8, 5)]

            goal_state = None

            distance = 1000000000
            for ghost in state.getGhostStates():
                pos = fixedGhostPos(ghost)

                if pos in spawn or state.getWalls()[int(pos[0])][int(pos[1])]:
                    continue  # ignore ghost spawn and walls
                manhattan = util.manhattanDistance(pos, state.getPacmanPosition())
                if manhattan > ghost.scaredTimer:
                    continue
                if manhattan < distance:
                    distance = manhattan
                    goal_state = pos

            if goal_state is None:
                return False

            return AStarToGoal(goal_state)

        def randomAction():
            return random.choice(state.getLegalActions())

        ourTree1 = BTSelector([
            BTSequence([
                BTLeaf(checkGhostsNearby),
                BTLeaf(fleeFromNearbyGhosts),
            ]),
            BTSequence([
                BTLeaf(checkForScaredGhosts),
                BTLeaf(aStarToNearestScaredGhost)
            ]),
            BTLeaf(aStarToNearestCapsule),
            BTLeaf(bfsToNearestPill)
        ])

        ourTree2 = BTSelector([
            BTSequence([
                BTLeaf(checkGhostsNearby),
                BTLeaf(aStarToNearestCapsule),
            ]),
            BTLeaf(fleeFromNearbyGhosts),
            BTSequence([
                BTLeaf(checkForScaredGhosts),
                BTLeaf(aStarToNearestScaredGhost)
            ]),
            BTLeaf(bfsToNearestPill)
        ])

        ourTree3 = BTSelector([
            BTLeaf(fleeFromNearbyGhosts),
            BTLeaf(randomAction)
        ])

        ourTree4 = BTLeaf(randomAction)

        action = ourTree3.evaluate()

        if not action:  # error handling basicly
            return 'Stop'
        else:
            return action


class MCTSAgent(Agent):
    def __init__(self):
        self.n = 30  # Depth of search
        self.max_time = 60.0 # how many seconds will we max use for a search
        self.Cp = 1
        self.tree = []
        self.turn = 0

    def getAction(self, state):
        self.tree = []

        self.turn += 1

        #                       0     1      2        3        4        5           6
        # root is id 0, node = [id ,state, parent, children, value, nr of visits, action]
        root = [0, state, 0, [], 0.0, 0, 'Stop']
        self.tree.append(root)

        self.start_time = time.time()
        i = 0

        while time.time() - self.start_time < self.max_time and i < self.n:
            i += 1
            print "Turn:", self.turn, "depth:", i
            node = self.selection(root)
            delta = self.simulation(node)
            self.backPropagate(node, delta)
        return self.bestResult(root)

    def selection(self, node):  # TreePolicy in pseudocode from slides
        state = node[1]
        tmp_node = node
        while not state.isWin() or not state.isLose():
            if time.time() - self.start_time > self.max_time + 10.0:  # if simulation is 10 seconds over time ->
                break                                                 # Stop it.
            if len(state.getLegalActions())-1 > len(tmp_node[3]):
                return self.expansion(tmp_node)
            else:
                bestChild = self.bestChild(tmp_node)
                if bestChild[0] is tmp_node[0]:
                    return self.expansion(tmp_node)
                tmp_node = bestChild
                state = tmp_node[1]
        return tmp_node

    def expansion(self, node):
        actions = ['Stop']
        for i in range(len(node[3])):
            childNode = self.tree[node[3][i]]
            actions.append(childNode[6])

        untried_actions = []

        for act in node[1].getLegalActions():
            if act not in actions:
                untried_actions.append(act)

        if len(untried_actions) is 0:
            return node

        act = random.choice(untried_actions)
        id = len(self.tree)

        child = [id, node[1].generatePacmanSuccessor(act), node[0], [], 0.0, 0, act]

        self.tree.append(child)
        self.tree[node[0]][3].append(id)

        return child

    def bestChild(self, node, Cp=None):
        if Cp is None:
            Cp = 1 / math.sqrt(2)  # Kocsis & Szepesvari

        highestUCT = -1000000
        child = 0

        if len(node[3]) is 0:
            return node

        for i in range(len(node[3])):
            childNode = self.tree[node[3][i]]
            sum = 0

            if len(childNode[3]) == 0:
                Xj = childNode[4]
            else:
                for j in range(len(childNode[3])):
                    sum += self.tree[childNode[3][j]][4]
                Xj = sum/len(childNode[3])

            div = 2.0 * math.log(len(self.tree)) / childNode[5]
            expl = math.sqrt(div)

            UCT = Xj + 2.0 * Cp * expl

            if UCT > highestUCT:
                highestUCT = UCT
                child = i

        return self.tree[node[3][child]]

    def simulation(self, node):  # DefaultPolicy
        state = copy.deepcopy(node[1])
        while not state.isWin() and not state.isLose():
            if time.time() - self.start_time > self.max_time + 10.0:  # if simulation is 10 seconds over time ->
                break                                                 # Stop it.
            act = random.choice(state.getLegalActions())
            state = state.generatePacmanSuccessor(act)
        return state.getScore()

    def backPropagate(self, node, delta):
        tmp_node = node
        while tmp_node is not None:
            tmp_node[5] += 1
            tmp_node[4] += delta
            self.tree[tmp_node[0]] = tmp_node

            if tmp_node[0] is 0:
                break

            tmp_node = self.tree[tmp_node[2]]

    def bestResult(self, node):
        return self.bestChild(node)[6]


class GAAgent(Agent):
    def __init__(self, genome=None):
        self.legal_composit = ["SEQ", "SEL"]
        self.legal_leaf = [
            "Go.North",
            "Go.East",
            "Go.South",
            "Go.West",
            "Valid.North",
            "Valid.East",
            "Valid.South",
            "Valid.West",
            "Danger.North",
            "Danger.East",
            "Danger.South",
            "Danger.West",
            # "Go.Random",
            # "GoNot.North",
            # "GoNot.East",
            # "GoNot.South",
            # "GoNot.West",
        ]
        self.legal_decorator = ["Invert"]
        self.legal_nodes = self.legal_composit + self.legal_leaf + self.legal_decorator

        # self.genome = ["SEL",
        #    ["SEQ", "Valid.North", "Danger.North", "GoNot.North"],
        #    ["SEQ", "Valid.East", "Danger.East", "GoNot.East"],
        #    ["SEQ", "Valid.South", "Danger.South", "GoNot.South"],
        #    ["SEQ", "Valid.West", "Danger.West", "GoNot.West"],
        #    "Go.Random"]

        if genome is None:
            self.genome = ["SEL", "Go.Stop"]
        else:
            self.genome = genome

        self.tree = GA_util.parse_node(self.genome, None)

    def copy(self):
        clone = GAAgent()
        clone.genome = copy.deepcopy(self.genome)
        return clone

    def print_genome(self):
        def print_help(genome, prefix=''):
            for gene in genome:
                if isinstance(gene, list):
                    print_help(gene, prefix + "  ")
                elif gene in self.legal_composit:
                    print prefix, gene
                else:
                    print prefix + '  ', gene

        print_help(self.genome)

    def mutate(self):
        """ YOUR CODE HERE! """
        mutate = random.choice([1, 2, 3, 4])
        if mutate is 1:
            # Add new leaf
            composit_ids = []
            for i in range(len(self.genome)):
                if self.genome[i] in self.legal_composit:
                    composit_ids.append(i)

            chosen_composit = random.choice[composit_ids]
            if type(self.genome[chosen_composit + 1]) == list:
                self.genome[chosen_composit + 1].append(random.choice(self.legal_leaf))
            else:
                self.genome[chosen_composit + 1] = [self.genome[chosen_composit + 1]]
                self.genome[chosen_composit + 1].append(random.choice(self.legal_leaf))

        # if mutate is 2:
        #     # Add new composite and leaf
        # if mutate is 3:
        #     # Change random composite or leaf to another composite or leaf
        # if mutate is 4:
        #     # Remove on leaf and composite if empty.
        if self.genome == ["SEL", "Go.Stop"]:
            self.genome = [random.choice(self.legal_composit), random.choice(self.legal_leaf)]
        return GAAgent(genome=self.genome)

    def getAction(self, state):
        action = self.tree(state)
        if action not in state.getLegalPacmanActions():
            # print "Illegal action!!"
            action = 'Stop'
        return action


#######################################################
# This portion is written for you, but will only work #
#       after you fill in parts of search.py          #
#######################################################

class SearchAgent(Agent):
    """
    This very general search agent finds a path using a supplied search
    algorithm for a supplied search problem, then returns actions to follow that
    path.

    As a default, this agent runs DFS on a PositionSearchProblem to find
    location (1,1)

    Options for fn include:
      depthFirstSearch or dfs
      breadthFirstSearch or bfs


    Note: You should NOT change any code in SearchAgent
    """

    def __init__(self, fn='depthFirstSearch', prob='PositionSearchProblem', heuristic='nullHeuristic'):
        # Warning: some advanced Python magic is employed below to find the right functions and problems

        # Get the search function from the name and heuristic
        if fn not in dir(search):
            raise AttributeError, fn + ' is not a search function in search.py.'
        func = getattr(search, fn)
        if 'heuristic' not in func.func_code.co_varnames:
            print('[SearchAgent] using function ' + fn)
            self.searchFunction = func
        else:
            if heuristic in globals().keys():
                heur = globals()[heuristic]
            elif heuristic in dir(search):
                heur = getattr(search, heuristic)
            else:
                raise AttributeError, heuristic + ' is not a function in searchAgents.py or search.py.'
            print('[SearchAgent] using function %s and heuristic %s' % (fn, heuristic))
            # Note: this bit of Python trickery combines the search algorithm and the heuristic
            self.searchFunction = lambda x: func(x, heuristic=heur)

        # Get the search problem type from the name
        if prob not in globals().keys() or not prob.endswith('Problem'):
            raise AttributeError, prob + ' is not a search problem type in SearchAgents.py.'
        self.searchType = globals()[prob]
        print('[SearchAgent] using problem type ' + prob)

    def registerInitialState(self, state):
        """
        This is the first time that the agent sees the layout of the game
        board. Here, we choose a path to the goal. In this phase, the agent
        should compute the path to the goal and store it in a local variable.
        All of the work is done in this method!

        state: a GameState object (pacman.py)
        """
        if self.searchFunction == None: raise Exception, "No search function provided for SearchAgent"
        starttime = time.time()
        problem = self.searchType(state)  # Makes a new search problem
        self.actions = self.searchFunction(problem)  # Find a path
        totalCost = problem.getCostOfActions(self.actions)
        print('Path found with total cost of %d in %.1f seconds' % (totalCost, time.time() - starttime))
        if '_expanded' in dir(problem): print('Search nodes expanded: %d' % problem._expanded)

    def getAction(self, state):
        """
        Returns the next action in the path chosen earlier (in
        registerInitialState).  Return Directions.STOP if there is no further
        action to take.

        state: a GameState object (pacman.py)
        """
        if 'actionIndex' not in dir(self): self.actionIndex = 0
        i = self.actionIndex
        self.actionIndex += 1
        if i < len(self.actions):
            return self.actions[i]
        else:
            return Directions.STOP


class PositionSearchProblem(search.SearchProblem):
    """
    A search problem defines the state space, start state, goal test, successor
    function and cost function.  This search problem can be used to find paths
    to a particular point on the pacman board.

    The state space consists of (x,y) positions in a pacman game.

    Note: this search problem is fully specified; you should NOT change it.
    """

    def __init__(self, gameState, costFn=lambda x: 1, goal=(1, 1), start=None, warn=True, visualize=True):
        """
        Stores the start and goal.

        gameState: A GameState object (pacman.py)
        costFn: A function from a search state (tuple) to a non-negative number
        goal: A position in the gameState
        """
        self.walls = gameState.getWalls()
        self.startState = gameState.getPacmanPosition()
        if start != None: self.startState = start
        self.goal = goal
        self.costFn = costFn
        self.visualize = visualize
        if warn and (gameState.getNumFood() != 1 or not gameState.hasFood(*goal)):
            print 'Warning: this does not look like a regular search maze'

        # For display purposes
        self._visited, self._visitedlist, self._expanded = {}, [], 0  # DO NOT CHANGE

    def getStartState(self):
        return self.startState

    def isGoalState(self, state):
        isGoal = state == self.goal

        # For display purposes only
        if isGoal and self.visualize:
            self._visitedlist.append(state)
            import __main__
            if '_display' in dir(__main__):
                if 'drawExpandedCells' in dir(__main__._display):  # @UndefinedVariable
                    __main__._display.drawExpandedCells(self._visitedlist)  # @UndefinedVariable

        return isGoal

    def getSuccessors(self, state):
        """
        Returns successor states, the actions they require, and a cost of 1.

         As noted in search.py:
             For a given state, this should return a list of triples,
         (successor, action, stepCost), where 'successor' is a
         successor to the current state, 'action' is the action
         required to get there, and 'stepCost' is the incremental
         cost of expanding to that successor
        """

        successors = []
        for action in [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]:
            x, y = state
            dx, dy = Actions.directionToVector(action)
            nextx, nexty = int(x + dx), int(y + dy)
            if not self.walls[nextx][nexty]:
                nextState = (nextx, nexty)
                cost = self.costFn(nextState)
                successors.append((nextState, action, cost))

        # Bookkeeping for display purposes
        self._expanded += 1  # DO NOT CHANGE
        if state not in self._visited:
            self._visited[state] = True
            self._visitedlist.append(state)

        return successors

    def getCostOfActions(self, actions):
        """
        Returns the cost of a particular sequence of actions. If those actions
        include an illegal move, return 999999.
        """
        if actions == None: return 999999
        x, y = self.getStartState()
        cost = 0
        for action in actions:
            # Check figure out the next state and see whether its' legal
            dx, dy = Actions.directionToVector(action)
            x, y = int(x + dx), int(y + dy)
            if self.walls[x][y]: return 999999
            cost += self.costFn((x, y))
        return cost


class StayEastSearchAgent(SearchAgent):
    """
    An agent for position search with a cost function that penalizes being in
    positions on the West side of the board.

    The cost function for stepping into a position (x,y) is 1/2^x.
    """

    def __init__(self):
        self.searchFunction = search.uniformCostSearch
        costFn = lambda pos: .5 ** pos[0]
        self.searchType = lambda state: PositionSearchProblem(state, costFn, (1, 1), None, False)


class StayWestSearchAgent(SearchAgent):
    """
    An agent for position search with a cost function that penalizes being in
    positions on the East side of the board.

    The cost function for stepping into a position (x,y) is 2^x.
    """

    def __init__(self):
        self.searchFunction = search.uniformCostSearch
        costFn = lambda pos: 2 ** pos[0]
        self.searchType = lambda state: PositionSearchProblem(state, costFn)


def manhattanHeuristic(position, problem, info={}):
    "The Manhattan distance heuristic for a PositionSearchProblem"
    xy1 = position
    xy2 = problem.goal
    return abs(xy1[0] - xy2[0]) + abs(xy1[1] - xy2[1])


def euclideanHeuristic(position, problem, info={}):
    "The Euclidean distance heuristic for a PositionSearchProblem"
    xy1 = position
    xy2 = problem.goal
    return ((xy1[0] - xy2[0]) ** 2 + (xy1[1] - xy2[1]) ** 2) ** 0.5


#####################################################
# This portion is incomplete.  Time to write code!  #
#####################################################

class CornersProblem(search.SearchProblem):
    """
    This search problem finds paths through all four corners of a layout.

    You must select a suitable state space and successor function
    """

    def __init__(self, startingGameState):
        """
        Stores the walls, pacman's starting position and corners.
        """
        self.walls = startingGameState.getWalls()
        self.startingPosition = startingGameState.getPacmanPosition()
        top, right = self.walls.height - 2, self.walls.width - 2
        self.corners = ((1, 1), (1, top), (right, 1), (right, top))
        for corner in self.corners:
            if not startingGameState.hasFood(*corner):
                print 'Warning: no food in corner ' + str(corner)
        self._expanded = 0  # DO NOT CHANGE; Number of search nodes expanded
        # Please add any code here which you would like to use
        # in initializing the problem
        "*** YOUR CODE HERE ***"

    def getStartState(self):
        """
        Returns the start state (in your state space, not the full Pacman state
        space)
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
        Returns whether this search state is a goal state of the problem.
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
        Returns successor states, the actions they require, and a cost of 1.

         As noted in search.py:
            For a given state, this should return a list of triples, (successor,
            action, stepCost), where 'successor' is a successor to the current
            state, 'action' is the action required to get there, and 'stepCost'
            is the incremental cost of expanding to that successor
        """

        successors = []
        for action in [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]:
            # Add a successor state to the successor list if the action is legal
            # Here's a code snippet for figuring out whether a new position hits a wall:
            #   x,y = currentPosition
            #   dx, dy = Actions.directionToVector(action)
            #   nextx, nexty = int(x + dx), int(y + dy)
            #   hitsWall = self.walls[nextx][nexty]

            "*** YOUR CODE HERE ***"

        self._expanded += 1  # DO NOT CHANGE
        return successors

    def getCostOfActions(self, actions):
        """
        Returns the cost of a particular sequence of actions.  If those actions
        include an illegal move, return 999999.  This is implemented for you.
        """
        if actions == None: return 999999
        x, y = self.startingPosition
        for action in actions:
            dx, dy = Actions.directionToVector(action)
            x, y = int(x + dx), int(y + dy)
            if self.walls[x][y]: return 999999
        return len(actions)


def cornersHeuristic(state, problem):
    """
    A heuristic for the CornersProblem that you defined.

      state:   The current search state
               (a data structure you chose in your search problem)

      problem: The CornersProblem instance for this layout.

    This function should always return a number that is a lower bound on the
    shortest path from the state to a goal of the problem; i.e.  it should be
    admissible (as well as consistent).
    """
    corners = problem.corners  # These are the corner coordinates
    walls = problem.walls  # These are the walls of the maze, as a Grid (game.py)

    "*** YOUR CODE HERE ***"
    return 0  # Default to trivial solution


class AStarCornersAgent(SearchAgent):
    "A SearchAgent for FoodSearchProblem using A* and your foodHeuristic"

    def __init__(self):
        self.searchFunction = lambda prob: search.aStarSearch(prob, cornersHeuristic)
        self.searchType = CornersProblem


class FoodSearchProblem:
    """
    A search problem associated with finding the a path that collects all of the
    food (dots) in a Pacman game.

    A search state in this problem is a tuple ( pacmanPosition, foodGrid ) where
      pacmanPosition: a tuple (x,y) of integers specifying Pacman's position
      foodGrid:       a Grid (see game.py) of either True or False, specifying remaining food
    """

    def __init__(self, startingGameState):
        self.start = (startingGameState.getPacmanPosition(), startingGameState.getFood())
        self.walls = startingGameState.getWalls()
        self.startingGameState = startingGameState
        self._expanded = 0  # DO NOT CHANGE
        self.heuristicInfo = {}  # A dictionary for the heuristic to store information

    def getStartState(self):
        return self.start

    def isGoalState(self, state):
        return state[1].count() == 0

    def getSuccessors(self, state):
        "Returns successor states, the actions they require, and a cost of 1."
        successors = []
        self._expanded += 1  # DO NOT CHANGE
        for direction in [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]:
            x, y = state[0]
            dx, dy = Actions.directionToVector(direction)
            nextx, nexty = int(x + dx), int(y + dy)
            if not self.walls[nextx][nexty]:
                nextFood = state[1].copy()
                nextFood[nextx][nexty] = False
                successors.append((((nextx, nexty), nextFood), direction, 1))
        return successors

    def getCostOfActions(self, actions):
        """Returns the cost of a particular sequence of actions.  If those actions
        include an illegal move, return 999999"""
        x, y = self.getStartState()[0]
        cost = 0
        for action in actions:
            # figure out the next state and see whether it's legal
            dx, dy = Actions.directionToVector(action)
            x, y = int(x + dx), int(y + dy)
            if self.walls[x][y]:
                return 999999
            cost += 1
        return cost


class AStarFoodSearchAgent(SearchAgent):
    "A SearchAgent for FoodSearchProblem using A* and your foodHeuristic"

    def __init__(self):
        self.searchFunction = lambda prob: search.aStarSearch(prob, foodHeuristic)
        self.searchType = FoodSearchProblem


def foodHeuristic(state, problem):
    """
    Your heuristic for the FoodSearchProblem goes here.

    This heuristic must be consistent to ensure correctness.  First, try to come
    up with an admissible heuristic; almost all admissible heuristics will be
    consistent as well.

    If using A* ever finds a solution that is worse uniform cost search finds,
    your heuristic is *not* consistent, and probably not admissible!  On the
    other hand, inadmissible or inconsistent heuristics may find optimal
    solutions, so be careful.

    The state is a tuple ( pacmanPosition, foodGrid ) where foodGrid is a Grid
    (see game.py) of either True or False. You can call foodGrid.asList() to get
    a list of food coordinates instead.

    If you want access to info like walls, capsules, etc., you can query the
    problem.  For example, problem.walls gives you a Grid of where the walls
    are.

    If you want to *store* information to be reused in other calls to the
    heuristic, there is a dictionary called problem.heuristicInfo that you can
    use. For example, if you only want to count the walls once and store that
    value, try: problem.heuristicInfo['wallCount'] = problem.walls.count()
    Subsequent calls to this heuristic can access
    problem.heuristicInfo['wallCount']
    """
    position, foodGrid = state
    "*** YOUR CODE HERE ***"
    return 0


class ClosestDotSearchAgent(SearchAgent):
    "Search for all food using a sequence of searches"

    def registerInitialState(self, state):
        self.actions = []
        currentState = state
        while (currentState.getFood().count() > 0):
            nextPathSegment = self.findPathToClosestDot(currentState)  # The missing piece
            self.actions += nextPathSegment
            for action in nextPathSegment:
                legal = currentState.getLegalActions()
                if action not in legal:
                    t = (str(action), str(currentState))
                    raise Exception, 'findPathToClosestDot returned an illegal move: %s!\n%s' % t
                currentState = currentState.generateSuccessor(0, action)
        self.actionIndex = 0
        print 'Path found with cost %d.' % len(self.actions)

    def findPathToClosestDot(self, gameState):
        """
        Returns a path (a list of actions) to the closest dot, starting from
        gameState.
        """
        # Here are some useful elements of the startState
        startPosition = gameState.getPacmanPosition()
        food = gameState.getFood()
        walls = gameState.getWalls()
        problem = AnyFoodSearchProblem(gameState)

        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()


class AnyFoodSearchProblem(PositionSearchProblem):
    """
    A search problem for finding a path to any food.

    This search problem is just like the PositionSearchProblem, but has a
    different goal test, which you need to fill in below.  The state space and
    successor function do not need to be changed.

    The class definition above, AnyFoodSearchProblem(PositionSearchProblem),
    inherits the methods of the PositionSearchProblem.

    You can use this search problem to help you fill in the findPathToClosestDot
    method.
    """

    def __init__(self, gameState):
        "Stores information from the gameState.  You don't need to change this."
        # Store the food for later reference
        self.food = gameState.getFood()

        # Store info for the PositionSearchProblem (no need to change this)
        self.walls = gameState.getWalls()
        self.startState = gameState.getPacmanPosition()
        self.costFn = lambda x: 1
        self._visited, self._visitedlist, self._expanded = {}, [], 0  # DO NOT CHANGE

    def isGoalState(self, state):
        """
        The state is Pacman's position. Fill this in with a goal test that will
        complete the problem definition.
        """
        x, y = state

        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()


def mazeDistance(point1, point2, gameState):
    """
    Returns the maze distance between any two points, using the search functions
    you have already built. The gameState can be any game state -- Pacman's
    position in that state is ignored.

    Example usage: mazeDistance( (2,4), (5,6), gameState)

    This might be a useful helper function for your ApproximateSearchAgent.
    """
    x1, y1 = point1
    x2, y2 = point2
    walls = gameState.getWalls()
    assert not walls[x1][y1], 'point1 is a wall: ' + str(point1)
    assert not walls[x2][y2], 'point2 is a wall: ' + str(point2)
    prob = PositionSearchProblem(gameState, start=point1, goal=point2, warn=False, visualize=False)
    return len(search.bfs(prob))


class MySuperAgent(Agent):
    def __init__(self):
        self.agent = SearchAgent(fn='aStarSearch', heuristic='manhattanHeuristic')

    def registerInitialState(self, state):
        return self.agent.registerInitialState(state)

    def getAction(self, state):
        return self.agent.getAction(state)
