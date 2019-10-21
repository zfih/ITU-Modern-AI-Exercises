import util


################################################################################
# This code section is the implementation of BFS, DFS and A* Search algorithms #
################################################################################

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    # get the current state from the problem
    state = problem.getStartState()

    # setup frontier as priority queue, with the priority being the cost to the state
    # thereby checking lowest cost first
    frontier = util.PriorityQueue()
    frontier.update(state, 0) # start state to frontier with 0 cost

    came_from = {}  # dictionary of what state we came from, indexed by position
    cost_so_far = {}  # dictionary of the cost so far, indexed by position
    came_from[state] = (None, 'Stop')  # put initial state in came_from
    cost_so_far[state] = 0  # put initial state in cost_so_far

    # while the frontier is not empty we keep going
    while not frontier.isEmpty():
        # pop lowest cost state and set to current_state
        current_state = frontier.pop()

        # if goal state is found, traverse the road backwards and return the reversed road
        if problem.isGoalState(current_state):
            road = []
            prev_pos, act = came_from[current_state]
            while prev_pos is not None:
                print act
                road.append(act)
                prev_pos, act = came_from[prev_pos]

            road.reverse()
            return road

        # for each successor of the current state set the cost to the successor as the cost so far
        # and the cost to the successor
        for next_state in problem.getSuccessors(current_state):
            new_cost = cost_so_far[current_state] + next_state[2]

            # if the next state is not in cost_so_far or the new cost is lower than previously found one
            if next_state[0] not in cost_so_far or new_cost < cost_so_far[next_state[0]]:
                cost_so_far[next_state[0]] = new_cost                       # then set the cost so far of the successor to the new cost
                priority = new_cost + heuristic(next_state[0], problem)     # set the priority to the new cost + the heuristic cost
                frontier.update(next_state[0], priority)                    # update(/push) frontier with new state and priority and
                came_from[next_state[0]] = (current_state, next_state[1])   # set the came_from of the successor to the current state and move on




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
    frontier = util.Stack()  # Use stack. We always want the latest element added.
    explored = []
    roads = {(-1, -1): []}  # Roads is a dictionary between a pos(x, y) and the road to get there as a list of actions.

    # Push initial state to frontier.
    frontier.push(problem.getStartState())

    # Keep going through frontier until empty.
    while not frontier.isEmpty():
        # pop the next state to check from frontier
        current_state = frontier.pop()

        # check if that new state is the goal state, if so return the road until this point
        if problem.isGoalState(current_state):
            return roads[current_state]

        # for each state we can get to from current state, generate it
        for state in problem.getSuccessors(current_state):
            # check if the states position is not already in frontier and not in explored
            # otherwise we already looked at it
            if state[0] not in frontier.list and state[0] not in explored:

                # if there is already a road to the current state, get a copy of it and assign it to road
                # otherwise road is empty
                if current_state in roads:
                    road = roads[current_state][:]
                else:
                    road = []

                # push the new state to the frontier to check its adjacent nodes later
                frontier.push(state[0])
                # append the action to get here to the road
                road.append(state[1])
                # and set the road of the state to the just updated road.
                roads[state[0]] = road[:]

        # finally add the state to the explored list so that we do not check it again
        explored.append(current_state)


def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""

    frontier = util.Queue() # Use Queue. We always want the oldest element in the frontier
    explored = []
    roads = {(-1, -1): []}  # Roads is a dictionary between a pos(x, y) and the road to get there as a list of actions.

    # Push initial state to frontier.
    frontier.push(problem.getStartState())

    # Keep going through frontier until empty.
    while not frontier.isEmpty():
        # pop the next state to check from frontier
        current_state = frontier.pop()

        # check if that new state is the goal state, if so return the road until this point
        if problem.isGoalState(current_state):
            return roads[current_state]

        # for each state we can get to from current state, generate it
        for state in problem.getSuccessors(current_state):
            # check if the states position is not already in frontier and not in explored
            # otherwise we already looked at it
            if state[0] not in frontier.list and state[0] not in explored:

                # if there is already a road to the current state, get a copy of it and assign it to road
                # otherwise road is empty
                if current_state in roads:
                    road = roads[current_state][:]
                else:
                    road = []

                # push the new state to the frontier to check its adjacent nodes later
                frontier.push(state[0])
                # append the action to get here to the road
                road.append(state[1])
                # and set the road of the state to the just updated road.
                roads[state[0]] = road[:]

        # finally add the state to the explored list so that we do not check it again
        explored.append(current_state)