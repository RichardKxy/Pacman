# multiAgents.py
# --------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
  """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
  """


  def getAction(self, gameState):
    """
    You do not need to change this method, but you're welcome to.

    getAction chooses among the best options according to the evaluation function.

    Just like in the previous project, getAction takes a GameState and returns
    some Directions.X for some X in the set {North, South, West, East, Stop}
    """
    # Collect legal moves and successor states
    legalMoves = gameState.getLegalActions()

    # Choose one of the best actions
    scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
    bestScore = max(scores)
    
    bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
    chosenIndex = random.choice(bestIndices) # Pick randomly among the best

    "Add more of your code here if you want to"

    return legalMoves[chosenIndex]

  def evaluationFunction(self, currentGameState, action):
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
    "*** YOUR CODE HERE ***"
    successorGameState = currentGameState.generatePacmanSuccessor(action)

    curPos = successorGameState.getPacmanPosition()
    curFood = successorGameState.getFood()
    curFoodlist = curFood.asList()

    curGhostStates = successorGameState.getGhostStates()
    curScaredTimes = [ghostState.scaredTimer for ghostState in curGhostStates]

    curGhostPos = successorGameState.getGhostPositions()

    CapsulesPos = successorGameState.getCapsules()
    walls = successorGameState.getWalls()

    score2Food = 0
    score = 0
    fooddist = []
    for food in curFoodlist :
      fooddist.append(util.manhattanDistance(curPos, food))
    if len(fooddist) != 0 :
      mindist = min(fooddist)
      score2Food = 1.0 / mindist - successorGameState.getNumFood()

    score2Ghost = 0
    ghostdist0cst = []
    ghostdist1cst = []
    score2Ghostlist = []
    for i in range(len(curGhostPos)) :
      if curScaredTimes[i] == 0 :
        ghostdist0cst.append( util.manhattanDistance(curPos, curGhostPos[i]))
      else :
        ghostdist1cst.append( (util.manhattanDistance(curPos, curGhostPos[i]), curScaredTimes[i]) )

    if len(ghostdist0cst) != 0 :
      mindist = min(ghostdist0cst)
      score2Ghost = mindist
      if mindist != 0 :
        score2Ghost = -1.0 / mindist
      else :
        score2Ghost = -5

    score2ghost = 0
    mindist = float("inf")
    if len(ghostdist1cst) != 0 : 
      for dist, cst in ghostdist1cst :
        if mindist > dist :
          mindist = dist
          if cst >= dist :
            if dist != 0 :
              score2ghost = 1.0 / dist
            else :
              score2ghost = 5
          else :
            if dist != 0 :
              score2ghost = -1.0 / dist
            else :
              score2ghost = -5

    score2Capsules = 0
    capsuleslist = []
    for capsules in CapsulesPos :
      capsuleslist.append(util.manhattanDistance(curPos, capsules))
    if len(capsuleslist) != 0 :
      mindist = min(capsuleslist)
      score2Capsules = 1.0 / mindist - len(capsuleslist)

    score = score2Food + score2Ghost + score2ghost + score2Capsules + currentGameState.getScore()

    return score

def scoreEvaluationFunction(currentGameState):
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

  def getAction(self, gameState):
    """
      Returns the minimax action from the current gameState using self.depth
      and self.evaluationFunction.

      Here are some method calls that might be useful when implementing minimax.

      gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

      Directions.STOP:
        The stop direction, which is always legal

      gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

      gameState.getNumAgents():
        Returns the total number of agents in the game
    """
    "*** YOUR CODE HERE ***"
    def maxValue(state, depth):
      depth += 1
      v = float('-Inf')
      
      if state.isWin() or state.isLose() or depth == self.depth:
        return self.evaluationFunction(state)
      
      for action in state.getLegalActions():
        v = max(v, minValue(state.generatePacmanSuccessor(action), depth, 1))
      return v

    def minValue(state, depth, numsofGhost):
      v = float('Inf')

      if state.isWin() or state.isLose():
        return self.evaluationFunction(state)

      for action in state.getLegalActions(numsofGhost):
        successorGameState = state.generateSuccessor(numsofGhost, action)
        if numsofGhost == (gameState.getNumAgents() - 1):          
          v = min(v, maxValue(successorGameState, depth))
        else:
          v = min(v, minValue(successorGameState, depth, numsofGhost + 1))
      return v

    legalMoves = gameState.getLegalActions()

    Max = float('-inf')
    for action in legalMoves : 
      if action != Directions.STOP :
        successorGameState = gameState.generatePacmanSuccessor(action)
        evaluate = minValue(successorGameState, 0, 1)
        if evaluate > Max :
          Max = evaluate
          move = action
    
    return move

class AlphaBetaAgent(MultiAgentSearchAgent):
  """
    Your minimax agent with alpha-beta pruning (question 3)
  """

  def getAction(self, gameState):
    """
      Returns the minimax action using self.depth and self.evaluationFunction
    """
    "*** YOUR CODE HERE ***"
    def maxValue(state, depth, alpha, beta):
      depth += 1
      v = float('-Inf')
      
      if state.isWin() or state.isLose() or depth == self.depth:
        return self.evaluationFunction(state)
      
      for action in state.getLegalActions():
        v = max(v, minValue(state.generatePacmanSuccessor(action), depth, 1, alpha, beta))
        alpha = max(alpha, v)
        if alpha >= beta :
          return v
      return v

    def minValue(state, depth, numsofGhost, alpha, beta):
      v = float('Inf')

      if state.isWin() or state.isLose():
        return self.evaluationFunction(state)

      for action in state.getLegalActions(numsofGhost):
        successorGameState = state.generateSuccessor(numsofGhost, action)
        if numsofGhost == (gameState.getNumAgents() - 1):          
          v = min(v, maxValue(successorGameState, depth, alpha, beta))
        else:
          v = min(v, minValue(successorGameState, depth, numsofGhost + 1, alpha, beta))
        beta = min(beta, v)
        if alpha >= beta :
          return v
      return v

    legalMoves = gameState.getLegalActions()

    Max = float('-inf')
    alpha = float('-inf')
    beta = float('inf')
    for action in legalMoves : 
      if action != Directions.STOP :
        successorGameState = gameState.generatePacmanSuccessor(action)
        evaluate = minValue(successorGameState, 0, 1, alpha, beta)
        if evaluate > Max :
          Max = evaluate
          move = action
    
    return move

class ExpectimaxAgent(MultiAgentSearchAgent):
  """
    Your expectimax agent (question 4)
  """

  def getAction(self, gameState):
    """
      Returns the expectimax action using self.depth and self.evaluationFunction

      All ghosts should be modeled as choosing uniformly at random from their
      legal moves.
    """
    "*** YOUR CODE HERE ***"
    def maxValue(state, depth):
      depth += 1
      v = float('-Inf')
      
      if state.isWin() or state.isLose() or depth == self.depth:
        return self.evaluationFunction(state)
      
      for action in state.getLegalActions():
        v = max(v, minValue(state.generatePacmanSuccessor(action), depth, 1))
      return v

    def minValue(state, depth, numsofGhost):
      v = float('Inf')

      if state.isWin() or state.isLose():
        return self.evaluationFunction(state)

      score = []
      for action in state.getLegalActions(numsofGhost):
        successorGameState = state.generateSuccessor(numsofGhost, action)
        if numsofGhost == (gameState.getNumAgents() - 1):          
          score.append(min(v, maxValue(successorGameState, depth)))
        else:
          score.append(min(v, minValue(successorGameState, depth, numsofGhost + 1)))
      if len(score) != 0 :
        result = 0
        for s in score:
          result += s
        v = result / len(score)
      return v

    legalMoves = gameState.getLegalActions()

    Max = float('-inf')
    for action in legalMoves : 
      if action != Directions.STOP :
        successorGameState = gameState.generatePacmanSuccessor(action)
        evaluate = minValue(successorGameState, 0, 1)
        if evaluate > Max :
          Max = evaluate
          move = action
    
    return move

def betterEvaluationFunction(currentGameState):
  """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    score2Food: find the smallest distance to pacman
    score2Ghost: find the samllest distance to the ghosts if their scaredtime is zero
    score2ghost: find the smallest distance to the ghosts if they are scared
    score2Capsules: get more close to the pellets
  """
  "*** YOUR CODE HERE ***"

  curPos = currentGameState.getPacmanPosition()
  curFood = currentGameState.getFood()
  curFoodlist = curFood.asList()

  curGhostStates = currentGameState.getGhostStates()
  curScaredTimes = [ghostState.scaredTimer for ghostState in curGhostStates]

  curGhostPos = currentGameState.getGhostPositions()

  CapsulesPos = currentGameState.getCapsules()
  walls = currentGameState.getWalls()

  score2Food = 0
  score = 0
  fooddist = []
  for food in curFoodlist :
    fooddist.append(util.manhattanDistance(curPos, food))
  if len(fooddist) != 0 :
    mindist = min(fooddist)
    score2Food = 1.0 / mindist - currentGameState.getNumFood()

  score2Ghost = 0
  ghostdist0cst = []
  ghostdist1cst = []
  score2Ghostlist = []
  for i in range(len(curGhostPos)) :
    if curScaredTimes[i] == 0 :
      ghostdist0cst.append( util.manhattanDistance(curPos, curGhostPos[i]))
    else :
      ghostdist1cst.append( (util.manhattanDistance(curPos, curGhostPos[i]), curScaredTimes[i]) )

  if len(ghostdist0cst) != 0 :
    mindist = min(ghostdist0cst)
    score2Ghost = mindist
    if mindist != 0 :
      score2Ghost = -1.0 / mindist
    else :
      score2Ghost = -5

  score2ghost = 0
  mindist = float("inf")
  if len(ghostdist1cst) != 0 : 
    for dist, cst in ghostdist1cst :
      if mindist > dist :
        mindist = dist
        if cst >= dist :
          if dist != 0 :
            score2ghost = 1.0 / dist
          else :
            score2ghost = 5
        else :
          if dist != 0 :
            score2ghost = -1.0 / dist
          else :
            score2ghost = -5

  score2Capsules = 0
  capsuleslist = []
  for capsules in CapsulesPos :
    capsuleslist.append(util.manhattanDistance(curPos, capsules))
  if len(capsuleslist) != 0 :
    mindist = min(capsuleslist)
    score2Capsules = 2.0 / mindist - len(capsuleslist)

  score = score2Food + score2Ghost + score2ghost + score2Capsules + currentGameState.getScore()

  return score

# Abbreviation
better = betterEvaluationFunction

class ContestAgent(MultiAgentSearchAgent):
  """
    Your agent for the mini-contest
  """

  def getAction(self, gameState):
    """
      Returns an action.  You can use any method you want and search to any depth you want.
      Just remember that the mini-contest is timed, so you have to trade off speed and computation.

      Ghosts don't behave randomly anymore, but they aren't perfect either -- they'll usually
      just make a beeline straight towards Pacman (or away from him if they're scared!)
    """
    "*** YOUR CODE HERE ***"
    def maxValue(state, depth):
      depth += 1
      v = float('-Inf')
      
      if state.isWin() or state.isLose() or depth == self.depth:
        return self.evaluate(state)
      
      for action in state.getLegalActions():
        v = max(v, minValue(state.generatePacmanSuccessor(action), depth, 1))
      return v

    def minValue(state, depth, numsofGhost):
      v = float('Inf')

      if state.isWin() or state.isLose():
        return self.evaluate(state)

      score = []
      for action in state.getLegalActions(numsofGhost):
        successorGameState = state.generateSuccessor(numsofGhost, action)
        if numsofGhost == (gameState.getNumAgents() - 1):          
          score.append(min(v, maxValue(successorGameState, depth)))
        else:
          score.append(min(v, minValue(successorGameState, depth, numsofGhost + 1)))
      if len(score) != 0 :
        result = 0
        for s in score:
          result += s
        v = result / len(score)
      return v

    legalMoves = gameState.getLegalActions()

    Max = float('-inf')
    for action in legalMoves : 
      if action != Directions.STOP :
        successorGameState = gameState.generatePacmanSuccessor(action)
        evaluate = minValue(successorGameState, 0, 1)
        if evaluate > Max :
          Max = evaluate
          move = action
    
    return move

  def evaluate(self, currentGameState) :
    curPos = currentGameState.getPacmanPosition()
    curFood = currentGameState.getFood()
    curFoodlist = curFood.asList()

    curGhostStates = currentGameState.getGhostStates()
    curScaredTimes = [ghostState.scaredTimer for ghostState in curGhostStates]
    curGhostPos = currentGameState.getGhostPositions()
    CapsulesPos = currentGameState.getCapsules()

    score2Food = 0
    score = 0
    fooddist = []
    for food in curFoodlist :
      fooddist.append(util.manhattanDistance(curPos, food))
    if len(fooddist) != 0 :
      mindist = min(fooddist)
      score2Food = 1.0 / mindist - currentGameState.getNumFood()

    score2Ghost = 0
    ghostdist0cst = []
    ghostdist1cst = []
    for i in range(len(curGhostPos)) :
      if curScaredTimes[i] == 0 :
        ghostdist0cst.append( util.manhattanDistance(curPos, curGhostPos[i]))
      else :
        ghostdist1cst.append( (util.manhattanDistance(curPos, curGhostPos[i]), curScaredTimes[i]) )

    if len(ghostdist0cst) != 0 :
      mindist = min(ghostdist0cst)
      if mindist != 0 :
        score2Ghost = -1.0 / mindist
      else :
        score2Ghost = -5

    score2ghost = 0
    mindist = float("inf")
    if len(ghostdist1cst) != 0 : 
      for dist, cst in ghostdist1cst :
        if mindist > dist :
          mindist = dist
          if cst >= dist :
            if dist != 0 :
              score2ghost = 1.0 / dist
            else :
              score2ghost = 5
          else :
            if dist != 0 :
              score2ghost = -1.0 / dist
            else :
              score2ghost = -5

    score2Capsules = 0
    capsuleslist = []
    for capsules in CapsulesPos :
      capsuleslist.append(util.manhattanDistance(curPos, capsules))
    if len(capsuleslist) != 0 :
      mindist = min(capsuleslist)
      score2Capsules = 1.0 / mindist - len(capsuleslist)

    score = score2Food + 1.5 * score2Ghost + score2ghost + score2Capsules #+currentGameState.getScore()

    return score

