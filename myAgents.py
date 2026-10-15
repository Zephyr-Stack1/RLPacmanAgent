import atexit
import csv
import os
import pickle
import random

from collections import defaultdict
from datetime import datetime

from game import Agent
from pacman import Directions


ALPHA = 0.2
GAMMA = 0.9
EPSILON = 0.3
EPISILON_DECAY = 0.995
MIN_EPSILON = 0.02
SAVE_INTERVAL = 50
TRAINING_NUM = 10000

"""
Use num to express the distance categories.
1 = one unit or less
2 = two units or less
3 = three units or less
4 = four units or farther
"""
DISTANCE_UNIT_1 = 1
DISTANCE_UNIT_2 = 2
DISTANCE_UNIT_3 = 3
DISTANCE_UNIT_4 = 4


STEP_PENALTY = -1
FOOD_REWARD = 100
CAPSULE_REWARD = 250
WIN_REWARD = 2000
LOSE_PENALTY = -2000

DANGER_DISTANCE = 4
MOVE_CLOSER_TO_DANGER_PENALTY = -20
MOVE_AWAY_FROM_DANGER_REWARD = 10
ADJACENT_DANGER_PENALTY = -100

MOVE_TOWARD_FOOD_REWARD = 4
MOVE_AWAY_FROM_FOOD_PENALTY = -2

MOVE_TOWARD_SCARED_GHOST_REWARD = 5

class TimidAgent(Agent):

    def __init__(
        self,
        alpha = ALPHA,
        gamma = GAMMA,
        epsilon = EPSILON,
        epsilonDecay = EPISILON_DECAY,
        minEpsilon = MIN_EPSILON,
        saveInterval = SAVE_INTERVAL,
        traningNum = TRAINING_NUM,
        qFile = "q_learning_training_v2.pkl",
        historyFile = "q_learning_training_history_v2.csv"
    ):
        super().__init__()

        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.epsilon = float(epsilon)
        self.epsilonDecay = float(epsilonDecay)
        self.minEpsilon = float(minEpsilon)

        self.traningNum = int(traningNum)
        self.saveInterval = int(saveInterval)

        # Unknown state-action pairs start at 0.0.
        self.qValues = defaultdict(float)

        self.episodesCompleted = 0

        # Counters that reset every time the program starts.
        self.gamesThisRun = 0
        self.trainingEpisodesThisRun = 0

        self.lastState = None
        self.lastAction = None
        self.lastScore = 0
        self.lastDangerDistance = None

        self.episodeSteps = 0
        self.episodeFoodEaten = 0
        self.episodeCapsulesEaten = 0

        self.trainingDataChanged = False

        self.runId = datetime.now().strftime(
            "%Y%m%d_%H%M%S_%f" )

        currentDirectory = os.path.dirname(
            os.path.abspath(__file__) )

        self.qFile = os.path.join( currentDirectory, qFile )

        self.historyFile = os.path.join(currentDirectory, historyFile )

        self.load_training_data()

        # current_epsilon = max(epsilon_min, epsilon * episodes ** n)
        if (self.alpha > 0 and self.epsilon > 0 and self.episodesCompleted > 0):
            tmpEpsilon = self.epsilon * ( self.epsilonDecay ** self.episodesCompleted)
            self.epsilon = max(self.minEpsilon, tmpEpsilon)


        atexit.register(self.save_training_data)
    def get_candidate_actions(self, state):

        legalActions = state.getLegalPacmanActions()

        nonStopActions = [
            action
            for action in legalActions
            if action != Directions.STOP
        ]

        if nonStopActions:
            return nonStopActions

        return legalActions

    def get_relative_direction(self, source, target):
        """
        Return relative direction as (horizontal, vertical).

        Horizontal:
            -1 = west
             0 = same column
             1 = east

        Vertical:
            -1 = south
             0 = same row
             1 = north
        """
        sourceX, sourceY= source
        targetX, targetY = target

        deltaX = targetX - sourceX
        deltaY = targetY - sourceY

        horizontal = 0
        vertical = 0

        if deltaX > 0:
            horizontal = 1
        elif deltaX < 0:
            horizontal = -1

        if deltaY > 0:
            vertical = 1
        elif deltaY < 0:
            vertical = -1

        return horizontal, vertical

    def get_distance_bucket(self, distance):

        if distance <= DISTANCE_UNIT_1:
            return DISTANCE_UNIT_1

        if distance <= DISTANCE_UNIT_2:
            return DISTANCE_UNIT_2

        if distance <= DISTANCE_UNIT_3:
            return DISTANCE_UNIT_3

        return DISTANCE_UNIT_4

    def get_absolute_distance(self, position, otherPosition):
        return abs(position[0] - otherPosition[0]) + abs(position[1] - otherPosition[1])

    def get_nearest_position_info(
        self,
        pacmanPosition,
        positions
    ):
        # delete all None from positions
        validPositions = [
            position
            for position in positions
            if position is not None
        ]

        if not validPositions:
            return (0, 0), DISTANCE_UNIT_4

        nearestPosition = min(
            validPositions,
            key = lambda position: self.get_absolute_distance(
            position,
            pacmanPosition)
        )

        distance = self.get_absolute_distance(nearestPosition, pacmanPosition)

        direction = self.get_relative_direction(pacmanPosition, nearestPosition)

        distanceBucket = self.get_distance_bucket(distance)

        return direction, distanceBucket

    def get_nearest_danger_distance(self, state):

        pacmanX, pacmanY = state.getPacmanPosition()

        distances = []

        for ghost in state.getGhostStates():
            if ghost.scaredTimer > 0:
                continue

            ghostPosition = ghost.getPosition()

            if ghostPosition is None:
                continue

            ghostX, ghostY = ghostPosition

            distance = ( abs(ghostX - pacmanX) + abs(ghostY - pacmanY))

            distances.append(distance)

        if not distances:
            return None

        return min(distances)

    def get_state_key(self, state):

        pacmanPosition = state.getPacmanPosition()
        ghostStates = state.getGhostStates()

        dangerousGhostPositions = [
            ghost.getPosition()
            for ghost in ghostStates
            if ghost.scaredTimer == 0
        ]

        scaredGhostPositions = [
            ghost.getPosition()
            for ghost in ghostStates
            if ghost.scaredTimer > 0
        ]

        foodPositions = state.getFood().asList()

        dangerDirection, dangerDistance = (
            self.get_nearest_position_info(
                pacmanPosition,
                dangerousGhostPositions
            )
        )

        scaredDirection, scaredDistance = (
            self.get_nearest_position_info(
                pacmanPosition,
                scaredGhostPositions
            )
        )

        foodDirection, foodDistance = (
            self.get_nearest_position_info(
                pacmanPosition,
                foodPositions
            )
        )

        legalActions = tuple(
            sorted(self.get_candidate_actions(state))
        )

        return (
            legalActions,
            dangerDirection,
            dangerDistance,
            scaredDirection,
            scaredDistance,
            foodDirection,
            foodDistance
        )

    def get_q_value(self, stateKey, action):
        return self.qValues[(stateKey, action)]

    def choose_action(self, stateKey, legalActions):

        if not legalActions:
            return Directions.STOP

        # Use epsilon-greedy to choose an action
        if random.random() < self.epsilon:
            return random.choice(legalActions)

        qValues = []

        for action in legalActions:
            qValue = self.get_q_value(stateKey, action)
            qValues.append(qValue)

        bestQValue = max(qValues)

        bestActions = []

        for action in legalActions:
            qValue = self.get_q_value(stateKey, action)

            if qValue == bestQValue:
                bestActions.append(action)

        return random.choice(bestActions)

    def update_sarsa(self, stateKey, action, reward, nextStateKey, nextAction):
        """
        SARSA update:

        Q(s,a) <- Q(s,a) + alpha * [reward + gamma * Q(s',a') - Q(s,a)]
        """

        # alpha = 0 means agent does not learn.
        if self.alpha == 0:
            return

        oldQValue = self.get_q_value(stateKey, action)

        if (nextStateKey is None or nextAction is None):
            target = reward
        else:
            nextQValue = self.get_q_value(nextStateKey, nextAction)

            target = (reward + self.gamma * nextQValue)

        # TD error
        tdError = target - oldQValue

        # Update Q(s, a)
        newQValue = (oldQValue + self.alpha * tdError)

        self.qValues[(stateKey, action)] = newQValue
        self.trainingDataChanged = True

    def update_q_learning(self, stateKey, action, reward, nextStateKey, nextLegalActions):
        """
        Q-learning update:

        Q(s,a) <- Q(s,a) + alpha *[reward + gamma * max Q(s',a') - Q(s,a)]
        """
        # alpha = 0 means agent does not learn.
        if self.alpha == 0:
            return

        oldQValue = self.get_q_value(stateKey, action)

        if (nextStateKey is None or not nextLegalActions):
            target = reward

        else:
            nextQValues = []
            for nextAction in nextLegalActions:
                nextQValue = self.get_q_value(nextStateKey, nextAction)
                nextQValues.append(nextQValue)

            bestNextQValue = max(nextQValues)

            target = (reward + self.gamma * bestNextQValue)

        tdError = target - oldQValue

        newQValue = (oldQValue + self.alpha * tdError)

        self.qValues[(stateKey, action)] = newQValue
        self.trainingDataChanged = True

    def update_sarsa(self, stateKey, action, reward, nextStateKey, nextAction):
        """
        SARSA:
        Q(s,a) <- Q(s,a) + alpha * (reward + gamma * Q(s',a') - Q(s,a))

        """

        if self.alpha == 0:
            return

        oldQValue = self.get_q_value(stateKey, action)

        if nextStateKey is None:
            target = reward
        else:
            nextQValue = self.get_q_value(nextStateKey, nextAction)

            target = (reward + self.gamma * nextQValue)

        tdError = target - oldQValue

        self.qValues[(stateKey, action)] = (oldQValue + self.alpha * tdError)

        self.trainingDataChanged = True


    def calculate_reward(self, previousState, currentState, terminal = False):
        """
        r_t = R_step
             + N_food * R_food
             + N_capsules * R_capsules
             + R_danger
             + R_adjacent
             + R_food_progress
             + R_scared_ghost
             + R_terminal
        """

        # Reward of step
        reward = STEP_PENALTY

        # Reward of food
        previousFoodCount = previousState.getNumFood()
        currentFoodCount = currentState.getNumFood()
        foodEaten = max(0, previousFoodCount - currentFoodCount)
        if foodEaten:
            reward += foodEaten * FOOD_REWARD
            self.episodeFoodEaten += foodEaten

        # Reward of capsules
        previousCapsuleCount = len(previousState.getCapsules())
        currentCapsuleCount = len(currentState.getCapsules())
        capsulesEaten = max(
            0, previousCapsuleCount - currentCapsuleCount
        )
        if capsulesEaten:
            reward += capsulesEaten * CAPSULE_REWARD
            self.episodeCapsulesEaten += capsulesEaten

        # Reward of danger
        previousDangerDistance = self.get_nearest_danger_distance(
            previousState
        )
        currentDangerDistance = self.get_nearest_danger_distance(
            currentState
        )

        if (
            previousDangerDistance is not None
            and currentDangerDistance is not None
        ):
            if (
                currentDangerDistance < previousDangerDistance
                and currentDangerDistance <= DANGER_DISTANCE
            ):
                reward += MOVE_CLOSER_TO_DANGER_PENALTY
            elif (
                currentDangerDistance > previousDangerDistance
                and previousDangerDistance <= DANGER_DISTANCE
            ):
                reward += MOVE_AWAY_FROM_DANGER_REWARD

        if (
            currentDangerDistance is not None
            and currentDangerDistance <= 1
            and not currentState.isLose()
        ):
            reward += ADJACENT_DANGER_PENALTY

        # Reward of food
        if foodEaten == 0:
            previousFoodPositions = previousState.getFood().asList()
            currentFoodPositions = currentState.getFood().asList()
            previousFoodDistance = None
            currentFoodDistance = None

            if previousFoodPositions:
                foodDistances = []
                for foodPosition in previousFoodPositions:
                    distance = self.get_absolute_distance(
                        previousState.getPacmanPosition(),
                        foodPosition
                    )
                    foodDistances.append(distance)
                previousFoodDistance = min(foodDistances)
            
            if currentFoodPositions:
                foodDistances = []
                for foodPosition in currentFoodPositions:
                    distance = self.get_absolute_distance(
                        currentState.getPacmanPosition(),
                        foodPosition
                    )
                    foodDistances.append(distance)
                currentFoodDistance = min(foodDistances)

            if (
                previousFoodDistance is not None
                and currentFoodDistance is not None
            ):
                if currentFoodDistance < previousFoodDistance:
                    reward += MOVE_TOWARD_FOOD_REWARD
                elif currentFoodDistance > previousFoodDistance:
                    reward += MOVE_AWAY_FROM_FOOD_PENALTY

        # Reward of scared ghosts
        previousScaredPositions = [
            ghost.getPosition()
            for ghost in previousState.getGhostStates()
            if ghost.scaredTimer > 0 and ghost.getPosition() is not None
        ]
        currentScaredPositions = [
            ghost.getPosition()
            for ghost in currentState.getGhostStates()
            if ghost.scaredTimer > 0 and ghost.getPosition() is not None
        ]
        previousScaredDistance = None
        currentScaredDistance = None
        if previousScaredPositions:
            ghostDistances = []
            for ghostPosition in previousScaredPositions:
                distance = self.get_absolute_distance(
                    previousState.getPacmanPosition(),
                    ghostPosition
                )
                ghostDistances.append(distance)
            previousScaredDistance = min(ghostDistances)
        if currentScaredPositions:
            ghostDistances = []
            for ghostPosition in currentScaredPositions:
                distance = self.get_absolute_distance(
                    currentState.getPacmanPosition(),
                    ghostPosition
                )
                ghostDistances.append(distance)
            currentScaredDistance = min(ghostDistances)
     
        if (
            previousScaredDistance is not None
            and currentScaredDistance is not None
            and currentScaredDistance < previousScaredDistance
        ):
            reward += MOVE_TOWARD_SCARED_GHOST_REWARD

        # Reward for winning or losing
        if terminal:
            if currentState.isWin():
                reward += WIN_REWARD
            elif currentState.isLose():
                reward += LOSE_PENALTY

        return reward


    def calculateReward_old(
        self,
        currentScore,
        currentDangerDistance
    ):
        reward = currentScore - self.lastScore

        if (
            self.lastDangerDistance is not None
            and currentDangerDistance is not None
        ):
            movedCloser = (
                currentDangerDistance
                < self.lastDangerDistance
            )

            movedFarther = (
                currentDangerDistance
                > self.lastDangerDistance
            )

            if (
                movedCloser
                and currentDangerDistance <= 4
            ):
                reward -= 2.0

            elif (
                movedFarther
                and self.lastDangerDistance <= 4
            ):
                reward += 1.0

        if (
            currentDangerDistance is not None
            and currentDangerDistance <= 1
        ):
            reward -= 10.0

        return reward

    def getAction(self, state):

        self.episodeSteps += 1

        currentState = self.get_state_key(state)
        #currentScore = state.getScore()

        currentDangerDistance = (self.get_nearest_danger_distance(state))

        legalActions = self.get_candidate_actions(state)

        if self.lastState is not None:
            reward = self.calculate_reward(
                self.lastState,
                state
            )

            self.update_q_learning(
                self.get_state_key(self.lastState),
                self.lastAction,
                reward,
                currentState,
                legalActions
            )

            self.lastScore += reward

        currentAction = self.choose_action(currentState, legalActions)

        self.lastState = state
        self.lastAction = currentAction
        #self.lastScore = currentScore
        self.lastDangerDistance = currentDangerDistance

        return currentAction

    def load_training_data(self):

        if not os.path.exists(self.qFile):
            print(
                "No saved Q-learning data found. "
                "It will Start a new training."
            )
            return

        try:
            with open(self.qFile, "rb") as file:
                trainingData = pickle.load(file)

            savedQValues = trainingData.get( "q_values", {})

            self.qValues.update(savedQValues)

            self.episodesCompleted = int(trainingData.get("episodes_completed", 0))

            print(
                "Loaded Q-learning training data:",
                len(self.qValues),
                "Q-Values",
                self.episodesCompleted,
                "total training episodes"
            )

        except (
            OSError,
            EOFError,
            ValueError,
            TypeError,
            AttributeError,
            pickle.UnpicklingError
        ) as error:
            print("Could not load Q-learning data:", error)

    def save_training_data(self):

        if not self.trainingDataChanged:
            return

        trainingData = {
            "version": 2,
            "q_values": dict(self.qValues),
            "episodes_completed": self.episodesCompleted,
            "saved_at": datetime.now().isoformat(
                timespec="seconds"
            ),
            "alpha": self.alpha,
            "gamma": self.gamma,
            "epsilon": self.epsilon,
            "numTrainingPerRun": self.traningNum
        }

        temporaryFile = self.qFile + ".tmp"

        try:
            with open(temporaryFile, "wb") as file:
                pickle.dump(
                    trainingData,
                    file,
                    protocol=pickle.HIGHEST_PROTOCOL
                )

            os.replace(
                temporaryFile,
                self.qFile
            )

            self.trainingDataChanged = False

            print(
                "Saved Q-learning training data:",
                len(self.qValues),
                "Q-values,",
                self.episodesCompleted,
                "total training episodes"
            )

        except OSError as error:
            print(
                "Could not save Q-learning data:",
                error
            )

    def recordEpisode(
        self,
        state,
        wasTraining,
        episodeAlpha,
        episodeEpsilon
    ):
        fieldNames = [
            "run_id",
            "run_episode",
            "training_episode",
            "mode",
            "timestamp",
            "q_value_count",
            "win",
            "steps",
            "epsilon",
            "alpha",
            "gamma",
            "score"
        ]

        fileHasContent = (
            os.path.exists(self.historyFile) and os.path.getsize( self.historyFile) > 0
        )

        row = {
            "run_id": self.runId,
            "run_episode": self.gamesThisRun,
            "training_episode": self.episodesCompleted,
            "mode": (
                "training"
                if wasTraining
                else "evaluation"
            ),
            "timestamp": datetime.now().isoformat(
                timespec="seconds"
            ),
            "q_value_count": len(self.qValues),     
            "win": state.isWin(),
            "steps": self.episodeSteps,
            "epsilon": episodeEpsilon,
            "alpha": episodeAlpha,
            "gamma": self.gamma,
            "score": self.lastScore
            #"score": round(self.last_score * 100)
        }

        try:
            with open(self.historyFile, "a", newline="", encoding="utf-8") as file:
                writer = csv.DictWriter(file, fieldnames = fieldNames)

                if not fileHasContent:
                    writer.writeheader()

                writer.writerow(row)

        except OSError as error:
            print("Could not record episode history:", error)


    def final(self, state):
        wasTraining = self.alpha > 0

        episodeAlpha = self.alpha
        episodeEpsilon = self.epsilon

        if self.lastState is not None:
            finalReward = self.calculate_reward(
                self.lastState,
                state,
                terminal = True
            )

            self.update_q_learning(
                self.get_state_key(self.lastState),
                self.lastAction,
                finalReward,
                None,
                None
            )

            self.lastScore += finalReward

        self.gamesThisRun += 1

        if wasTraining:
            self.episodesCompleted += 1
            self.trainingEpisodesThisRun += 1

            self.epsilon = max(
                self.minEpsilon,
                self.epsilon
                * self.epsilonDecay
            )

        reachedTrainingLimit = (
            self.traningNum > 0
            and self.trainingEpisodesThisRun
            >= self.traningNum
        )

        if reachedTrainingLimit:
            self.alpha = 0.0
            self.epsilon = 0.0

        self.recordEpisode(
            state,
            wasTraining,
            episodeAlpha,
            episodeEpsilon
        )

        shouldSave = (
            wasTraining
            and (
                self.episodesCompleted
                % self.saveInterval == 0
                or reachedTrainingLimit
            )
        )

        if shouldSave:
            self.save_training_data()

        self.lastState = None
        self.lastAction = None
        self.lastScore = 0
        self.lastDangerDistance = None
        self.episodeSteps = 0
        self.episodeFoodEaten = 0
        self.episodeCapsulesEaten = 0
