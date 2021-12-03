import pygame
import random
from enum import Enum
from collections import namedtuple
import numpy as np

pygame.init()
font = pygame.font.Font('../singleplayer_snake/arial.ttf', 12)


class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4


Point = namedtuple('Point', 'x, y')

# rgb colors
WHITE = (255, 255, 255)
RED1 = (200, 0, 0)
RED2 = (255, 100, 0)
BLUE1 = (0, 0, 255)
BLUE2 = (0, 100, 255)
GREEN1 = (0, 200, 0)
GREEN2 = (100, 255, 100)
YELLOW1 = (200, 200, 0)
YELLOW2 = (255, 255, 100)
BLACK = (0, 0, 0)

BLOCK_SIZE = 20
SPEED = 9

KILL_MULT = 10
DEATH_MULT = 5
WIN_MULT = 10
START_MULT = 0
TIME_MULT = 50

W = 640//2
H = 480//2


class Snake:

    def __init__(self, pos, length=6, w=W, h=H):
        """
        initialises snake class
        :param x: x starting coordinate of head
        :param y: y starting coordinate of head
        :param dir: starting direction
        :param length: length of snake
        """
        self.punished = False
        self.w = w  # width of game screen
        self.h = h  # height of game screen
        self.START_POS = [(self.w / 4, self.h / 4), (self.w / 4, self.h * 3 / 4), (3 * self.w / 4, self.h / 4), (3 * self.w / 4, self.h * 3 / 4)]
        self.pos = pos
        self.head = Point(self.START_POS[self.pos][0], self.START_POS[self.pos][1])
        if pos < 2:
            self.direction = Direction.RIGHT
        else:
            self.direction = Direction.LEFT
        self.reward = START_MULT  # starting reward
        self.alive = True  # True if snake is alive
        self.kills = 0  # number of kills snake has
        self.length = length
        # creating the snakes body
        if self.direction == Direction.RIGHT:
            self.body = [self.head]
            for i in range(length - 1):
                self.body.append(Point(self.head.x - (i + 1) * BLOCK_SIZE, self.head.y))
        else:
            self.body = [self.head]
            for i in range(length - 1):
                self.body.append(Point(self.head.x + (i + 1) * BLOCK_SIZE, self.head.y))

    def move(self, action):
        """
        determines if the snake should go straight, left or right
        :param action: Value signifying the snakes movement
        """
        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clock_wise.index(self.direction)

        if np.array_equal(action, [1, 0, 0]):
            new_dir = clock_wise[idx]  # straight
        elif np.array_equal(action, [0, 1, 0]):
            next_idx = (idx + 1) % 4
            new_dir = clock_wise[next_idx]  # right turn
        else:
            next_idx = (idx - 1) % 4
            new_dir = clock_wise[next_idx]  # left turn
        self.direction = new_dir

        x = self.head.x
        y = self.head.y
        if self.direction == Direction.RIGHT:
            x += BLOCK_SIZE
        elif self.direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif self.direction == Direction.DOWN:
            y += BLOCK_SIZE
        elif self.direction == Direction.UP:
            y -= BLOCK_SIZE
        self.head = Point(x, y)

    def is_collision(self, index, snakes, frame_iteration,  pt=None):
        """
        Detects if snake is dead or not
        :param index: index of snake in list
        :param snakes: list of snakes
        :param pt: The point to check if collision
        :return: True if collision
        """
        if pt is None:
            pt = self.head

        if len(self.body) != 0 or self.alive:
            # hits boundary
            if pt.x > self.w - BLOCK_SIZE or pt.x < 0 or pt.y > self.h - BLOCK_SIZE or pt.y < 0:
                return True

            # hits itself
            if pt in self.body[1:] and frame_iteration > 5:
                return True

            # hits another snake
            for i in range(len(snakes)):
                if pt in snakes[i].body and i != index:
                    snakes[i].kills += 1
                    snakes[i].reward += KILL_MULT
                    return True

        return False


class SnakeGame:
    def __init__(self, snakes, w=W, h=H):
        self.w = w  # width of game screen
        self.h = h  # height of game screen
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('Snake')
        self.clock = pygame.time.Clock()
        self.snakes = snakes
        self.num_snakes = len(snakes)
        self.frame_iteration = 0

    def play_step(self, actions):
        # 1. collect user input
        self.frame_iteration += 1
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        # 2. move
        count = 0
        for i in range(len(self.snakes)):
            snake = self.snakes[i]
            if not snake.alive:
                count += 1
            index = i - count
            if snake.alive:
                snake.body.insert(0, snake.head)
                snake.move(actions[index])  # update the head
                snake.body.pop()  # update the tail

        # 3. update screen
        self._update_ui()

        # 4. check if snakes are alive
        for i in range(len(self.snakes)):
            if self.snakes[i].is_collision(i, self.snakes, self.frame_iteration):
                self.snakes[i].alive = False
                self.num_snakes -= 1

        for snake in self.snakes:
            if not snake.alive and snake.punished == False:
                snake.reward -= (DEATH_MULT + self.num_snakes)
                snake.body = []
                snake.punished = True
                # if self.frame_iteration < 8:
                #     snake.reward -= 10
                #elif self.frame_iteration < 10:
                 #   snake.reward -= 10
                # if 10 < self.frame_iteration <= 15:
                #     snake.reward += 10
                # elif self.frame_iteration > 15:
                #     snake.reward += 25
            elif snake.alive:
                snake.reward += int(self.frame_iteration)*0.001

        # 5. check if game is are over
        if self.num_snakes == 1:
            for snake in self.snakes:
                if snake.alive:
                    snake.reward += WIN_MULT  # rewarding winner
            game_over = True
        elif self.num_snakes == 0 or self.frame_iteration > 75:
            game_over = True
        else:
            game_over = False

        # 6. update clock
        self.clock.tick(SPEED)

        # 7. return game over
        return [self.snakes[0].reward, self.snakes[1].reward, self.snakes[2].reward, self.snakes[3].reward], game_over

    def _update_ui(self):
        self.display.fill(BLACK)

        if self.snakes[0].alive:  # print player 1
            for pt in self.snakes[0].body:
                pygame.draw.rect(self.display, BLUE1, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
                pygame.draw.rect(self.display, BLUE2, pygame.Rect(pt.x + 4, pt.y + 4, 12, 12))
        if self.snakes[1].alive:  # print player 2
            for pt in self.snakes[1].body:
                pygame.draw.rect(self.display, RED1, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
                pygame.draw.rect(self.display, RED2, pygame.Rect(pt.x + 4, pt.y + 4, 12, 12))
        if len(self.snakes) > 2 and self.snakes[2].alive:  # print player 3
            for pt in self.snakes[2].body:
                pygame.draw.rect(self.display, GREEN1, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
                pygame.draw.rect(self.display, GREEN2, pygame.Rect(pt.x + 4, pt.y + 4, 12, 12))
        if len(self.snakes) > 3 and self.snakes[3].alive:  # print player 4
            for pt in self.snakes[3].body:
                pygame.draw.rect(self.display, YELLOW1, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
                pygame.draw.rect(self.display, YELLOW2, pygame.Rect(pt.x + 4, pt.y + 4, 12, 12))

        # printing rewards
        # text = 'Player1 reward = ' + str(self.snakes[0].reward) + '  |  Player2 reward = ' + str(self.snakes[1].reward)
        # if len(self.snakes) > 2:
        #     text += '  |  Player3 reward = ' + str(self.snakes[2].reward)
        # if len(self.snakes) > 3:
        #     text += '  |  Player4 reward = ' + str(self.snakes[3].reward)
        # self.display.blit(font.render(text, True, WHITE), [0, 0])
        pygame.display.flip()

    def reset(self):
        # init game state
        self.frame_iteration = 0
        self.num_snakes = len(self.snakes)
        for snake in self.snakes:
            # update starting point
            snake.punished = False
            snake.pos = (snake.pos + 1) % 4
            snake.head = Point(snake.START_POS[snake.pos][0], snake.START_POS[snake.pos][1])
            # update direction
            if snake.pos < 2:
                snake.direction = Direction.RIGHT
            else:
                snake.direction = Direction.LEFT
            snake.reward = START_MULT
            snake.alive = True
            snake.kills = 0
            if snake.direction == Direction.RIGHT:
                snake.body = [snake.head]
                for i in range(snake.length - 1):
                    snake.body.append(Point(snake.head.x - (i + 1) * BLOCK_SIZE, snake.head.y))
            else:
                snake.body = [snake.head]
                for i in range(snake.length - 1):
                    snake.body.append(Point(snake.head.x + (i + 1) * BLOCK_SIZE, snake.head.y))

# if __name__ == '__main__':
#     player1 = Snake(w / 4, h / 4, Direction.RIGHT)
#     player2 = Snake(w / 4, h * 3 / 4, Direction.RIGHT)
#     player3 = Snake(3 * w / 4, h / 4, Direction.LEFT)
#     player4 = Snake(3 * w / 4, h * 3 / 4, Direction.LEFT)
#     players = [player1, player2, player3]
#     game = SnakeGame(players)
#
#     # game loop
#     while True:
#         game_over = game.play_step()
#
#         if game_over == True:
#             break
#
#     print('Player1 reward = ' + str(player1.reward) + "\nPlayer2 reward = " + str(player2.reward))
#     if len(players) > 2:
#         print('Player3 reward = ' + str(player3.reward))
#     if len(players) > 3:
#         print('Player4 reward = ' + str(player4.reward))
#     pygame.quit()
