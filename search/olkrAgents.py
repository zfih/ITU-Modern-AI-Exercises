from game import Directions
from game import Agent
from game import Actions

import random
import util
import time
import search
import math


#############################################################
# This code section is the implementation of Behavior Trees #
#############################################################

# Superclass for nodes in the Behavior Tree
class BTNode:
    def __init__(self, children):
        self.children = children


# Sequence node
class BTSequence(BTNode):
    def evaluate(self):
        """
        Evaluate each childrens function and return the result.
        If a node fails we stop and return False, otherwise we return
        the last result that a child returned.
        """

        for node in self.children:
            result = node.evaluate()
            if not result:
                return False
        return result


# Selector node
class BTSelector(BTNode):
    def evaluate(self):
        """
        Evaluate each childrens function and return the result.
        If a node succeeds we return the result of their function.
        If each node fails we return a False for fail.
        """
        for node in self.children:
            result = node.evaluate()
            if result:
                return result
        return False


# Leaf node for BT
# Can both be a condition node and action node depending on function implementation
class BTLeaf(BTNode):
    def __init__(self, function, params=None):
        """
        Setup function and parameters if necessary.
        """
        self.function = function
        self.params = params

    def evaluate(self):
        """
        Evaluate each function with params if any exist and return
        the result to parent.
        """
        if self.params is None:
            return self.function()
        return self.function(self.params)


class BTDecorator(BTNode):
    def __init__(self, child, function):
        """
        Setup function and child.
        """
        self.children = child
        self.function = function

    def evaluate(self):
        """
        Evaluate given function that takes a BTNode as argument
        which should then be evaluated if the function decides to.
        The function should return the result of the child function
        if it decides to call it.
        """
        return self.function(self.children)


def fixedGhostPos(ghost):
    """
    Function for fixing ghost positions for A* and similar when ghosts are scared.

    Since the ghosts move half steps when scared, we can remove the difference of the half
    step depending on the direction the ghost just went in.
    This means we are effectively aiming for the spot behind the ghost whenever there is a half step,
    but in practice we eat them in the next step or when we hit them when moving toward them.
    We are retracting to make sure that we don't accidently target a wall.
    """
    pos = ghost.getPosition()
    dir = ghost.getDirection()  # last direction the ghost went.

    # if we are not at half-step, just return pos
    if pos[0] % 1.0 != 0.5 and pos[1] % 1.0 != 0.5:
        return pos

    # otherwise we fix the pos and return
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
    """
    Takes the a state of the game and checks each possible state and ghost state
    two actions into the future and returns true if there is any of those states that
    result in the death of the agent, otherwise False.
    """
    for act in state.getLegalActions():
        # Set current state and add to list of possible states.
        # then for each action from current state, generate the next pacman state
        possibleStates = [state]
        next_state = state.generatePacmanSuccessor(act)

        # As ghosts apparently are not moved by generatePacmanSuccessor, we
        # generate a new state for each of actions for each ghost
        for ghostAct in next_state.getLegalActions(1):
            for ghost2Act in next_state.getLegalActions(2):
                # generate first temp state using ghost action for ghost one
                tmp_state1 = next_state.generateSuccessor(1, ghostAct)
                if tmp_state1.isLose():  # if we lose return true
                    return True

                # generate second temp state using ghost action for ghost two
                tmp_state2 = tmp_state1.generateSuccessor(2, ghost2Act)
                if tmp_state2.isLose():  # if we lose return true
                    return True

                # append final state to list of possible states.
                possibleStates.append(tmp_state2)

        # continue doing the same as before but for each state in the possible states
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

    return False  # if none of the states kill us, we have no ghosts near


class BTAgent(Agent):
    def getAction(self, state):
        """
        The get action function of the BTAgent that returns the action to be taken at the current state.
        """

        def fleeFromNearbyGhosts():
            """
            Similar to the function that checks if a ghost is near, but instead of returning true, it collects
            all roads in a list and all bad directions in another list, and then picks the first element in a road,
            not using a bad direction.
            """
            badDirections = ['Stop']
            ghostNearby = False

            roads = []

            # for all legal actions in current state
            for act in state.getLegalActions():
                possibleStates = [state]  # add current state to a possible state for later.
                next_state = state.generatePacmanSuccessor(act)  # begin generating next state
                road = [act]  # add action to road

                # As ghosts apparently are not moved by generatePacmanSuccessor, we
                # generate a new state for each of actions for each ghost
                for ghostAct in next_state.getLegalActions(1):
                    for ghost2Act in next_state.getLegalActions(2):
                        # generate first temp state using ghost action for ghost one
                        tmp_state1 = next_state.generateSuccessor(1, ghostAct)
                        if tmp_state1.isLose():  # if we lose we add the direction to bad and continue checking
                            ghostNearby = True
                            badDirections.append(act)
                            continue
                        # generate second temp state using ghost action for ghost two
                        tmp_state2 = tmp_state1.generateSuccessor(2, ghost2Act)
                        if tmp_state2.isLose():
                            ghostNearby = True
                            badDirections.append(act)

                        # append final state to list of possible states.
                        possibleStates.append(tmp_state2)

                # continue doing the same as before but for each state in the possible states
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

            # if we do find a ghost nearby, shuffle the roads and pick the first one that is good
            # if no good ones exist, we will fall through and return false, we are dying no matter what
            if ghostNearby:
                random.shuffle(roads)
                for road in roads:
                    if road[0] not in badDirections:
                        return road[0]

            return False  # if no ghosts are nearby return false

        def checkGhostsNearby():
            # run generic check with current state
            return checkGhostsNearbyState(state)

        def bfsToNearestPill():
            """Use BFS to find nearest pill"""

            frontier = util.Queue()  # Use Queue. We always want the oldest element in the frontier
            explored = []
            roads = {}  # Roads is a dictionary between a pos(x, y) and the road to get there as a list of actions.

            states = {}  # states is a dictionary to a state from a pos
            states[state.getPacmanPosition()] = state  # add current state to list of states

            frontier.push(state.getPacmanPosition())  # add current state pos to frontier.

            # Keep going through frontier until empty.
            while not frontier.isEmpty():
                current_pos = frontier.pop()  # get new pos
                current_state = states[current_pos]  # get the state at that pos
                explored.append(current_pos)  # add pos to explored

                # otherwise keep searching with bfs
                for act in current_state.getLegalActions():
                    # generate new state and pos, and save these
                    next_state = current_state.generatePacmanSuccessor(act)
                    next_pos = next_state.getPacmanPosition()
                    states[next_pos] = next_state

                    # check if the states position is not already in frontier and not in explored
                    # otherwise we already looked at it
                    if next_pos not in frontier.list and next_pos not in explored:

                        # if there is already a road to the current state, get a copy of it and assign it to road
                        # otherwise road is empty
                        if current_pos in roads:
                            road = roads[current_pos][:]
                        else:
                            road = []

                        # push the new state to the frontier to check its adjacent nodes later
                        frontier.push(next_pos)
                        # append the action to get here to the road
                        road.append(act)
                        # and set the road of the state to the just updated road.
                        roads[next_pos] = road[:]

                        # if there is food at current pos then return the road to current pos
                        if current_state.getFood()[current_pos[0]][current_pos[1]]:  # how does duck-typing fail this?
                            return roads[current_pos][0]

            # return false if no pills are left. Should never happen as we would have won
            return False

        def checkForScaredGhosts():
            """
            Check if there are any scared ghosts, and if they are within distance of getting killed
            before they are no longer scared.
            """
            ghosts = state.getGhostStates()
            killableGhost = False

            # check both ghosts if their scared timer is 0
            for ghost in ghosts:
                if ghost.scaredTimer is 0:
                    continue
                # if the timer is not zero, get the distance to the ghosts
                distance = util.manhattanDistance(ghost.getPosition(), state.getPacmanPosition())

                # if distance to ghost is less than the time steps to get to ghost it is killable
                if distance < ghost.scaredTimer:
                    killableGhost = True

            if killableGhost:
                return True
            return False

        def AStarToGoal(goal):
            # Modified version from search.py

            """
            Search the node that has the lowest combined cost and heuristic first.
            """

            # setup frontier as priority queue, with the priority being the cost to the state
            # thereby checking lowest cost first
            frontier = util.PriorityQueue()
            frontier.update(state.getPacmanPosition(), 0)

            came_from = {}  # dictionary of what state we came from, indexed by position
            cost_so_far = {}  # dictionary of the cost so far, indexed by position
            came_from[state.getPacmanPosition()] = (None, 'Stop')  # put initial state in came_from
            cost_so_far[state.getPacmanPosition()] = 0  # put initial state in cost_so_far

            states = {}  # states is a dictionary to a state from a pos
            states[state.getPacmanPosition()] = state  # add current state to list of states

            # keep going until we find the goal or there is no more search space
            while not frontier.isEmpty():
                current_pos = frontier.pop()  # pop best position

                # if we are at our goal, backtrack the road
                if current_pos == goal:
                    road = []
                    prev_pos, act = came_from[current_pos]
                    while prev_pos is not None:
                        road.append(act)
                        prev_pos, act = came_from[prev_pos]
                    return road.pop()  # pop the last element as road is currently reversed.

                # get current state from current pos
                current_state = states[current_pos]

                # generate new states using legal actions
                for act in current_state.getLegalActions():
                    # for each successor of the current state set the cost to the successor as the cost so far
                    # and the cost to the successor
                    next_state = current_state.generatePacmanSuccessor(act)
                    states[next_state.getPacmanPosition()] = next_state
                    new_cost = cost_so_far[current_state.getPacmanPosition()] + 1  # it always costs 1 to move

                    # if the next state is not in cost_so_far or the new cost is lower than previously found one
                    if next_state.getPacmanPosition() not in cost_so_far or new_cost < cost_so_far[
                        next_state.getPacmanPosition()]:
                        # then set the cost so far of the successor to the new cost
                        cost_so_far[next_state.getPacmanPosition()] = new_cost
                        # set the priority to the new cost + manhattan distance to goal
                        priority = new_cost + util.manhattanDistance(goal, next_state.getPacmanPosition())
                        # update(/push) frontier with new state and priority and
                        frontier.update(next_state.getPacmanPosition(), priority)
                        # set the came_from of the successor to the current state and move on
                        came_from[next_state.getPacmanPosition()] = (current_pos, act)
            # if no road is found, return false
            return False

        def aStarToNearestCapsule():
            """
            Finds nearest capsule using Manhattan distance and A* to that location
            If no capsules are left, we return False
            """
            if len(state.getCapsules()) == 0:
                return False

            goal_state = None

            # find nearest possible goal_state
            distance = 1000000000
            for cap in state.getCapsules():
                manhattan = util.manhattanDistance(cap, state.getPacmanPosition())
                if manhattan < distance:
                    distance = manhattan
                    goal_state = cap

            # if something goes wrong just take a random capsule
            if goal_state is None:
                goal_state = random.choice(state.getCapsules())

            return AStarToGoal(goal_state)

        def aStarToNearestScaredGhost():
            """
            Finds nearest scared ghost and attacks them if they are within killing range, i.e
            they are still scared when reached.
            """
            if len(state.getGhostStates()) == 0:
                return False

            # set ghost spawn coords so we can ignore them,
            # they are dumb and dangerous as the ghosts respawn in there.
            spawn = [(11, 5), (10, 5), (10, 6), (9, 6), (9, 5), (8, 5)]

            goal_state = None

            # find neaest scared ghost not in spawn and in distance.
            distance = 1000000000
            for ghost in state.getGhostStates():
                pos = fixedGhostPos(ghost)

                if pos in spawn or state.getWalls()[int(pos[0])][int(pos[1])]:
                    continue  # ignore ghost spawn and walls
                manhattan = util.manhattanDistance(pos, state.getPacmanPosition())
                if manhattan > ghost.scaredTimer:  # if the distance is larger than the timer, we have no chance
                    continue
                if manhattan < distance:
                    distance = manhattan
                    goal_state = pos

            if goal_state is None:
                return False

            # A* to ghost
            return AStarToGoal(goal_state)

        def randomAction():
            # return random action.
            return random.choice(state.getLegalActions())


        # ourTree1, Graph of this in report
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

        # ourTree2, Graph of this in report
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

        # ourTree3, flee from nearby ghosts if any, otherwise random
        ourTree3 = BTSelector([
            BTLeaf(fleeFromNearbyGhosts),
            BTLeaf(randomAction)
        ])
        # ourTree4, random agent
        ourTree4 = BTLeaf(randomAction)

        action = ourTree2.evaluate()

        if not action:  # if no action is found, stand still
            return 'Stop'
        else:
            return action


######################################################################
# This code section is the implementation of Monte-Carlo Tree Search #
######################################################################

class MCTSAgent(Agent):
    def __init__(self):
        self.n = 30  # Depth of search
        self.max_time = 60.0  # how many seconds will we max use for a search
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
                break  # Stop it.
            if len(state.getLegalActions()) - 1 > len(tmp_node[3]):
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
                Xj = sum / len(childNode[3])

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
                break  # Stop it.
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
