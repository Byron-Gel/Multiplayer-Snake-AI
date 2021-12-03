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

    def __init__(self, mod):
        self.n_games = 0
        self.epsilon = 0  # randomness
        self.gamma = 0.95  # discount rate
        self.memory = deque(maxlen=MAX_MEMORY)  # popleft()
        self.mod = mod
        if mod == 0:  # just direction snake is going and if there is any danger close by
            self.model = Linear_QNet(13, 13*10, 3, 2)  # 7
        elif mod == 1:  # same as 0 but direction of opponents heads
            self.model = Linear_QNet(15, 15*10, 3, 2)  # 13
        elif mod == 2:  # same as 0 but direction of its own tail
            self.model = Linear_QNet(19, 19*10, 3, 2)  # 9
        else:  # 1 and 2 combined
            self.model = Linear_QNet(23, 23*10, 3, 2)  # 15
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)

    def get_head_dist(self, snakes, index, state):
        snake = snakes[index]
        if index != 0:
            if snakes[0].alive:
                state.append(1/(np.cosh((snakes[0].head.x - snake.head.x) / 20))) # distance of head in x direction
                state.append(1/(np.cosh((snakes[0].head.y - snake.head.x) / 20))) # distance of head in y direction
                # state.append(snakes[0].head.x < snake.head.x)  # head left
                # state.append(snakes[0].head.x > snake.head.x)  # head right
                # state.append(snakes[0].head.y < snake.head.y)  # head up
                # state.append(snakes[0].head.y > snake.head.y)  # head down
            else:
                for _ in range(2):
                    state.append(0)  # 0 if snake is dead
        if index != 1:
            if snakes[1].alive:
                state.append(1/(np.cosh((snakes[1].head.x - snake.head.x) / 20)))
                state.append(1/(np.cosh((snakes[1].head.y - snake.head.x) / 20)))
            else:
                for _ in range(2):
                    state.append(0)
        if index != 2:
            if snakes[2].alive:
                state.append(1/(np.cosh((snakes[2].head.x - snake.head.x) / 20)))
                state.append(1/(np.cosh((snakes[2].head.y - snake.head.x) / 20)))
            else:
                for _ in range(2):
                    state.append(0)
        if index != 3:
            if snakes[3].alive:
                state.append(1/(np.cosh((snakes[3].head.x - snake.head.x) / 20)))
                state.append(1 / (np.cosh((snakes[3].head.y - snake.head.x) / 20)))
            else:
                for _ in range(2):
                    state.append(0)

        return state

    def loc(self, snakes, index, state):
        snake = snakes[index]
        if index != 0:
            if snakes[0].alive:
                state.append(snakes[0].head.x < snake.head.x)  # head left
                state.append(snakes[0].head.x > snake.head.x)  # head right
                state.append(snakes[0].head.y < snake.head.y)  # head up
                state.append(snakes[0].head.y > snake.head.y)  # head down
            else:
                for _ in range(4):
                    state.append(0)  # 0 if snake is dead
        if index != 1:
            if snakes[1].alive:
                state.append(snakes[1].head.x < snake.head.x)  # head left
                state.append(snakes[1].head.x > snake.head.x)  # head right
                state.append(snakes[1].head.y < snake.head.y)  # head up
                state.append(snakes[1].head.y > snake.head.y)  # head down
            else:
                for _ in range(4):
                    state.append(0)
        if index != 2:
            if snakes[2].alive:
                state.append(snakes[1].head.x < snake.head.x)  # head left
                state.append(snakes[1].head.x > snake.head.x)  # head right
                state.append(snakes[1].head.y < snake.head.y)  # head up
                state.append(snakes[1].head.y > snake.head.y)  # head down
            else:
                for _ in range(4):
                    state.append(0)
        if index != 3:
            if snakes[3].alive:
                state.append(snakes[1].head.x < snake.head.x)  # head left
                state.append(snakes[1].head.x > snake.head.x)  # head right
                state.append(snakes[1].head.y < snake.head.y)  # head up
                state.append(snakes[1].head.y > snake.head.y)  # head down
            else:
                for _ in range(4):
                    state.append(0)

        return state

    def get_tail1(self, snakes, index, state):
        snake = snakes[index]
        if snake.alive:
            # state.append(snake.body[len(snake.body)-1].x < snake.head.x)  # tail left
            # state.append(snake.body[len(snake.body)-1].x > snake.head.x)  # tail right
            # state.append(snake.body[len(snake.body)-1].y < snake.head.y)  # tail up
            # state.append(snake.body[len(snake.body)-1].y > snake.head.y)  # tail down
            state.append(1/(np.cosh((snake.body[len(snake.body) - 1].x - snake.head.x) / 20)))
            state.append(1/(np.cosh((snake.body[len(snake.body) - 1].y - snake.head.y) / 20)))
        else:
            for _ in range(2):
                state.append(0)
        return state

    def get_tail2(self, snakes, index, state):
        snake = snakes[index]
        if snake.alive:
            state.append(snake.body[len(snake.body)-1].x < snake.head.x)  # tail left
            state.append(snake.body[len(snake.body)-1].x > snake.head.x)  # tail right
            state.append(snake.body[len(snake.body)-1].y < snake.head.y)  # tail up
            state.append(snake.body[len(snake.body)-1].y > snake.head.y)  # tail down
            # state.append(1/(np.cosh((snake.body[len(snake.body) - 1].x - snake.head.x) / 20)))
            # state.append(1/(np.cosh((snake.body[len(snake.body) - 1].y - snake.head.y) / 20)))
        else:
            for _ in range(2):
                state.append(0)
        return state

    def get_state(self, snakes, index, game):
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

        state = [
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

            # Move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,
        ]
        # add position of opposition heads if mod is 1 or 3
        if self.mod == 0 or self.mod == 1:
            self.get_head_dist(snakes, index, state)
        # # add position of players tail if mod is 2 or 3
        if self.mod == 1:
            self.get_tail1(snakes, index, state)

        if self.mod == 2 or self.mod == 3:
            self.loc(snakes, index, state)

        if self.mod == 3:
            self.get_tail2(snakes, index, state)



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
    agent = [Agent(0), Agent(1), Agent(2), Agent(3)]
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
                    plot_mean[i].append(sum(scores10[i][-n-1:-1]) / n)


            plot(plot_mean[0], plot_mean[1], plot_mean[2], plot_mean[3])

            game.reset()


if __name__ == '__main__':
    train()
