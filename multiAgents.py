# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent
from pacman import GameState

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState: GameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"
       # Pick less threatening action and donot get stuck for more than 3 moves
        #chosenIndex =  bestIndices[0]

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState: GameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        # use distance to food and distance to food instead of food position, evaluating state action pairs
        # use distance to ghost to avoid ghost
        # use scared time to eat ghost
        # use score to evaluate state
        # use distance to food to evaluate state
        # use distance to ghost to evaluate state
        # use scared time to evaluate state
        newFoodList = newFood.asList()
        foodDistances = [manhattanDistance(newPos, food) for food in newFoodList]
        GhostDistances = [manhattanDistance(newPos, ghost.getPosition()) for ghost in newGhostStates]
        
        print(str(newGhostStates))


        

            


        return (successorGameState.getScore() + 1/(min(foodDistances, default= 0) + 1) - 1/(min(GhostDistances) + 1) + sum(newScaredTimes))

def scoreEvaluationFunction(currentGameState: GameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        
        # get legal actions for pacman
        legalActions = gameState.getLegalActions(0)
        # get the number of ghosts
        numGhosts = gameState.getNumAgents() - 1
        # get the number of legal actions for pacman
        numLegalActions = len(legalActions)
        # initialize the best action to None
        bestAction = None
        # initialize the best score to negative infinity
        bestScore = float('-inf')
        # iterate over the legal actions for pacman
        for action in legalActions:
            # get the successor state for the action
            successorState = gameState.generateSuccessor(0, action)
            # get the score for the successor state
            score = self.minimax(successorState, 1, numGhosts, self.depth)
            # if the score is greater than the best score
            if score > bestScore:
                # update the best score
                bestScore = score
                # update the best action
                bestAction = action
        # return the best action
        return bestAction
    
        util.raiseNotDefined()
    def minimax(self, gameState, agentIndex, numGhosts, depth):
        # if the game state is a win or a loss or the depth is 0
        if gameState.isWin() or gameState.isLose() or depth == 0:
            # return the score of the game state
            return self.evaluationFunction(gameState)
        # if the agent index is 0
        if agentIndex == 0:
            # get the legal actions for pacman
            legalActions = gameState.getLegalActions(agentIndex)
            # initialize the best score to negative infinity
            bestScore = float('-inf')
            # iterate over the legal actions for pacman
            for action in legalActions:
                # get the successor state for the action
                successorState = gameState.generateSuccessor(agentIndex, action)
                # get the score for the successor state
                score = self.minimax(successorState, 1, numGhosts, depth)
                # update the best score
                bestScore = max(bestScore, score)
            # return the best score
            return bestScore
        # if the agent index is not 0
        else:
            # get the legal actions for the ghost
            legalActions = gameState.getLegalActions(agentIndex)
            # initialize the best score to positive infinity
            bestScore = float('inf')
            # if the agent index is the last ghost
            if agentIndex == numGhosts:
                # iterate over the legal actions for the ghost
                for action in legalActions:
                    # get the successor state for the action
                    successorState = gameState.generateSuccessor(agentIndex, action)
                    # get the score for the successor state
                    score = self.minimax(successorState, 0, numGhosts, depth - 1)
                    # update the best score
                    bestScore = min(bestScore, score)
            # if the agent index is not the last ghost
            else:
                # iterate over the legal actions for the ghost
                for action in legalActions:
                    # get the successor state for the action
                    successorState = gameState.generateSuccessor(agentIndex, action)
                    # get the score for the successor state
                    score = self.minimax(successorState, agentIndex + 1, numGhosts, depth)
                    # update the best score
                    bestScore = min(bestScore, score)
            # return the best score
            return bestScore
        
        util.raiseNotDefined()


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        """The minimax should expand with min for each number of ghost and max for pacman state and prune according to alpha beta"""
        # get legal actions for pacman
        legalActions = gameState.getLegalActions(0)
        # get the number of ghosts
        numGhosts = gameState.getNumAgents() - 1
        # get the number of legal actions for pacman
        numLegalActions = len(legalActions)
        # initialize the best action to None
        bestAction = None
        # initialize the best score to negative infinity
        bestScore = float('-inf')
        # initialize alpha to negative infinity
        alpha = float('-inf')
        # initialize beta to positive infinity
        beta = float('inf')
        # iterate over the legal actions for pacman
        for action in legalActions:
            # get the successor state for the action
            successorState = gameState.generateSuccessor(0, action)
            # get the score for the successor state
            score = self.minimax(successorState, 1, numGhosts, self.depth, alpha, beta)
            # if the score is greater than the best score
            if score > bestScore:
                # update the best score
                bestScore = score
                # update the best action
                bestAction = action
            # update alpha
            alpha = max(alpha, bestScore)
        # return the best action
        return bestAction

        util.raiseNotDefined()
    def minimax(self, gameState, agentIndex, numGhosts, depth, alpha, beta):
        # if the game state is a win or a loss or the depth is 0
        if gameState.isWin() or gameState.isLose() or depth == 0:
            # return the score of the game state
            return self.evaluationFunction(gameState)
        
        # To prune tree expansion based on alpha and beta
        if agentIndex == 0:
            # get the legal actions for pacman
            legalActions = gameState.getLegalActions(agentIndex)
            # initialize the best score to negative infinity
            bestScore = float('-inf')
            # iterate over the legal actions for pacman
            for action in legalActions:
                # get the successor state for the action
                successorState = gameState.generateSuccessor(agentIndex, action)
                # get the score for the successor state
                score = self.minimax(successorState, 1, numGhosts, depth, alpha, beta)
                # update the best score
                bestScore = max(bestScore, score)
                # if the best score is greater than beta
                if bestScore > beta:
                    # return the best score
                    return bestScore
                # update alpha
                alpha = max(alpha, bestScore)
            # return the best score
            return bestScore
        else:
            # get the legal actions for the ghost
            legalActions = gameState.getLegalActions(agentIndex)
            # initialize the best score to positive infinity
            bestScore = float('inf')
            # if the agent index is the last ghost
            if agentIndex == numGhosts:
                # iterate over the legal actions for the ghost
                for action in legalActions:
                    # get the successor state for the action
                    successorState = gameState.generateSuccessor(agentIndex, action)
                    # get the score for the successor state
                    score = self.minimax(successorState, 0, numGhosts, depth - 1, alpha, beta)
                    # update the best score
                    bestScore = min(bestScore, score)
                    # if the best score is less than alpha
                    if bestScore < alpha:
                        # return the best score
                        return bestScore
                    # update beta
                    beta = min(beta, bestScore)
            # if the agent index is not the last ghost
            else:
                # iterate over the legal actions for the ghost
                for action in legalActions:
                    # get the successor state for the action
                    successorState = gameState.generateSuccessor(agentIndex, action)
                    # get the score for the successor state
                    score = self.minimax(successorState, agentIndex + 1, numGhosts, depth, alpha, beta)
                    # update the best score
                    bestScore = min(bestScore, score)
                    # if the best score is less than alpha
                    if bestScore < alpha:
                        # return the best score
                        return bestScore
                    # update beta
                    beta = min(beta, bestScore)
            # return the best score
            return bestScore
        util.raiseNotDefined()
        

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        # get legal actions for pacman
        legalActions = gameState.getLegalActions(0)
        # get the number of ghosts
        numGhosts = gameState.getNumAgents() - 1
        # get the number of legal actions for pacman
        numLegalActions = len(legalActions)
        # initialize the best action to None
        bestAction = None
        # initialize the best score to negative infinity
        bestScore = float('-inf')
        # iterate over the legal actions for pacman
        for action in legalActions:
            # get the successor state for the action
            successorState = gameState.generateSuccessor(0, action)
            # get the score for the successor state
            score = self.expectimax(successorState, 1, numGhosts, self.depth)
            # if the score is greater than the best score
            if score > bestScore:
                # update the best score
                bestScore = score
                # update the best action
                bestAction = action
        # return the best action
        return bestAction
        util.raiseNotDefined()
    def expectimax(self, gameState, agentIndex, numGhosts, depth):
        # if the game state is a win or a loss or the depth is 0
        if gameState.isWin() or gameState.isLose() or depth == 0:
            # return the score of the game state
            return self.evaluationFunction(gameState)
        # if the agent index is 0
        if agentIndex == 0:
            # get the legal actions for pacman
            legalActions = gameState.getLegalActions(agentIndex)
            # initialize the best score to negative infinity
            bestScore = float('-inf')
            # iterate over the legal actions for pacman
            for action in legalActions:
                # get the successor state for the action
                successorState = gameState.generateSuccessor(agentIndex, action)
                # get the score for the successor state
                score = self.expectimax(successorState, 1, numGhosts, depth)
                # update the best score
                bestScore = max(bestScore, score)
            # return the best score
            return bestScore
        # if the agent index is not 0
        else:
            # get the legal actions for the ghost
            legalActions = gameState.getLegalActions(agentIndex)
            # initialize the best score to 0
            bestScore = 0
            # if the agent index is the last ghost
            if agentIndex == numGhosts:
                # iterate over the legal actions for the ghost
                for action in legalActions:
                    # get the successor state for the action
                    successorState = gameState.generateSuccessor(agentIndex, action)
                    # get the score for the successor state
                    score = self.expectimax(successorState, 0, numGhosts, depth - 1)
                    # update the best score
                    bestScore += score
            # if the agent index is not the last ghost
            else:
                # iterate over the legal actions for the ghost
                for action in legalActions:
                    # get the successor state for the action
                    successorState = gameState.generateSuccessor(agentIndex, action)
                    # get the score for the successor state
                    score = self.expectimax(successorState, agentIndex + 1, numGhosts, depth)
                    # update the best score
                    bestScore += score
            # return the best score
            return bestScore / len(legalActions)
        util.raiseNotDefined()



def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    # get the position of pacman
    pacmanPosition = currentGameState.getPacmanPosition()
    # get the food grid
    foodGrid = currentGameState.getFood()
    # get the food list
    foodList = foodGrid.asList()
    # get the ghost states
    ghostStates = currentGameState.getGhostStates()
    # get the scared times
    scaredTimes = [ghostState.scaredTimer for ghostState in ghostStates]
    # get the number of food
    numFood = len(foodList)
    # get the number of ghosts
    numGhosts = len(ghostStates)
    # get the score
    score = currentGameState.getScore()
    # get the distance to the nearest food
    distanceToNearestFood = min([manhattanDistance(pacmanPosition, food) for food in foodList], default=0)
    # get the distance to the nearest ghost
    distanceToNearestGhost = min([manhattanDistance(pacmanPosition, ghost.getPosition()) for ghost in ghostStates], default=0)
    # get the scared time
    scaredTime = sum(scaredTimes)
    # return the evaluation function
    return score + 1/(distanceToNearestFood + 1) - 1/(distanceToNearestGhost + 1) + scaredTime
    
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction
