import torch
import random
import numpy as np
from collections import deque
from gameAI import SnakeGame, Direction, Point, Snake
from models import Linear_QNet, QTrainer

from helper import plot

MAX_MEMORY = 50_000
BATCH_SIZE = 1000
LR = 0.001


class Agent:

    def __init__(self, numLayers = 2, getDirection=True, getFood=True, getDanger=True, getHeadDist=False, getLocation=False,
                 getTail=False, numSnake = 4):
        self.getDirection = getDirection
        self.getFood = getFood
        self.getDanger = getDanger
        self.getHeadDist = getHeadDist
        self.getLocation = getLocation
        self.getTail = getTail
        self.n_games = 0
        self.epsilon = 0  # randomness
        self.gamma = 0.95  # discount rate
        self.memory = deque(maxlen=MAX_MEMORY)  # popleft()
        inputSize = 4 * int(getDirection) + 4 * int(getFood) + 3 * int(getDanger) + 2 * (numSnake-1) * int(getHeadDist) + 4 * (numSnake-1) * int(
            getLocation) + 2 * int(getTail)
        self.model = Linear_QNet(inputSize, inputSize, 3, numLayers)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)

    def get_head_dist(self, snakes, index, state):
        snake = snakes[index]
        for i in range(len(snakes)):
            if i == index:
                continue
            if snakes[i].alive:
                state.append(1 / (np.cosh((snakes[i].head.x - snake.head.x) / 20)))  # distance of head in x direction
                state.append(1 / (np.cosh((snakes[i].head.y - snake.head.x) / 20)))  # distance of head in y direction
            else:
                for _ in range(2):
                    state.append(0)  # 0 if snake is dead
        return state

    def get_loc(self, snakes, index, state):
        snake = snakes[index]
        for i in range(len(snakes)):
            if i == index:
                continue
            if snakes[i].alive:
                state.append(snakes[i].head.x < snake.head.x)  # head left
                state.append(snakes[i].head.x > snake.head.x)  # head right
                state.append(snakes[i].head.y < snake.head.y)  # head up
                state.append(snakes[i].head.y > snake.head.y)  # head down
            else:
                for _ in range(4):
                    state.append(0)  # 0 if snake is dead

        return state

    def get_tail(self, snakes, index, state):
        snake = snakes[index]
        if snake.alive:
            state.append(1 / (np.cosh((snake.body[len(snake.body) - 1].x - snake.head.x) / 20)))
            state.append(1 / (np.cosh((snake.body[len(snake.body) - 1].y - snake.head.y) / 20)))
        else:
            for _ in range(2):
                state.append(0)
        return state

    def get_food(self, snakes, index, game, state):
        snake = snakes[index]
        # Food location
        foodPos = [
            game.food.x < snake.head.x,  # food left
            game.food.x > snake.head.x,  # food right
            game.food.y < snake.head.y,  # food up
            game.food.y > snake.head.y  # food down
        ]
        return state + foodPos

    def get_direction(self, snakes, index, state):
        snake = snakes[index]

        dir_l = snake.direction == Direction.LEFT
        dir_r = snake.direction == Direction.RIGHT
        dir_u = snake.direction == Direction.UP
        dir_d = snake.direction == Direction.DOWN

        return state + [dir_l, dir_r, dir_u, dir_d]

    def get_danger(self, snakes, index, game, state):
        snake = snakes[index]
        head = snake.body[0]
        point_l = Point(head.x - 20, head.y)
        point_r = Point(head.x + 20, head.y)
        point_u = Point(head.x, head.y - 20)
        point_d = Point(head.x, head.y + 20)

        dir_l = snake.direction == Direction.LEFT
        dir_r = snake.direction == Direction.RIGHT
        dir_u = snake.direction == Direction.UP
        dir_d = snake.direction == Direction.DOWN

        danger = [
            # Danger straight
            ((dir_r and snake.is_collision(index, snakes, game.frame_iteration, point_r)) or
             (dir_l and snake.is_collision(index, snakes, game.frame_iteration, point_l)) or
             (dir_u and snake.is_collision(index, snakes, game.frame_iteration, point_u)) or
             (dir_d and snake.is_collision(index, snakes, game.frame_iteration, point_d))),

            # Danger right
            ((dir_u and snake.is_collision(index, snakes, game.frame_iteration, point_r)) or
             (dir_d and snake.is_collision(index, snakes, game.frame_iteration, point_l)) or
             (dir_l and snake.is_collision(index, snakes, game.frame_iteration, point_u)) or
             (dir_r and snake.is_collision(index, snakes, game.frame_iteration, point_d))),

            # Danger left
            ((dir_d and snake.is_collision(index, snakes, game.frame_iteration, point_r)) or
             (dir_u and snake.is_collision(index, snakes, game.frame_iteration, point_l)) or
             (dir_r and snake.is_collision(index, snakes, game.frame_iteration, point_u)) or
             (dir_l and snake.is_collision(index, snakes, game.frame_iteration, point_d))),
        ]
        return state + danger

    def get_state(self, snakes, index, game):
        state = []
        if self.getDirection:
            state = self.get_direction(snakes, index, state)
        if self.getDanger:
            state = self.get_danger(snakes, index, game, state)
        if self.getFood:
            state = self.get_food(snakes, index, game, state)
        if self.getHeadDist:
            state = self.get_head_dist(snakes, index, state)
        if self.getLocation:
            state = self.get_loc(snakes, index, state)
        if self.getTail:
            state = self.get_tail(snakes, index, state)

        return np.array(state, dtype=int)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))  # popleft if MAX_MEMORY is reached

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)  # list of tuples
        else:
            mini_sample = self.memory

        for state, action, reward, next_state, done in mini_sample:
            self.trainer.train_step(state, action, reward, next_state, done)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        # random moves: tradeoff exploration / exploitation
        self.epsilon = 100 - self.n_games
        final_move = [0, 0, 0]
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        return final_move


def train():
    plot_mean = [[], [], [], []]
    scores10 = [[], [], [], []]
    player1 = Snake(0)
    player2 = Snake(1)
    player3 = Snake(2)
    player4 = Snake(3)
    players = [player1, player2, player3, player4]
    game = SnakeGame(players)
    agent = [Agent(), Agent(getHeadDist=True), Agent(getLocation=True), Agent(getTail=True)]
    while True:
        old_states = []
        final_moves = []
        for i in range(len(players)):
            # get old state
            if players[i].alive:
                state_old = agent[i].get_state(players, i, game)
                old_states.append(state_old)
                # get move
                final_moves.append(agent[i].get_action(state_old))
            else:
                final_moves.append(None)
                old_states.append(None)

        # perform move and get new state
        rewards, done = game.play_step(final_moves)
        new_states = []
        for i in range(len(players)):
            if players[i].alive:
                new_states.append(agent[i].get_state(players, i, game))
            else:
                new_states.append(None)

        # train short memory
        for i in range(len(players)):
            if players[i].alive:
                agent[i].train_short_memory(old_states[i], final_moves[i], rewards[i], new_states[i], done)

        # remember
        for i in range(len(players)):
            if players[i].alive:
                agent[i].remember(old_states[i], final_moves[i], rewards[i], new_states[i], done)

        if done:
            # train long memory, plot result
            for a in agent:
                a.n_games += 1
                a.train_long_memory()

            print('Player1 reward = ' + str(player1.reward) + "\nPlayer2 reward = " + str(player2.reward))
            if len(players) > 2:
                print('Player3 reward = ' + str(player3.reward))
            if len(players) > 3:
                print('Player4 reward = ' + str(player4.reward))
            rewards.sort(reverse=True)
            n = 10
            for i in range(len(players)):
                player = players[i]
                scores10[i].append(rewards.index(player.reward) + 1)
                if len(scores10[i]) < n:
                    plot_mean[i].append(sum(scores10[i]) / agent[i].n_games)
                else:
                    plot_mean[i].append(sum(scores10[i][-n - 1:-1]) / n)

            plot(plot_mean[0], plot_mean[1], plot_mean[2], plot_mean[3])

            game.reset()


if __name__ == '__main__':
    train()
