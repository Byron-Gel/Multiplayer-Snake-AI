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
SPEED = 6

KILL_MULT = 3
DEATH_MULT = 5
WIN_MULT = 10

w = 640
h = 480


class Snake:
    def __init__(self, x, y, dir):
        self.direction = dir
        self.head = Point(x, y)
        self.reward = 0
        self.alive = True
        self.kills = 0
        if dir == Direction.RIGHT:
            self.body = [self.head,
                         Point(self.head.x - BLOCK_SIZE, self.head.y),
                         Point(self.head.x - (2 * BLOCK_SIZE), self.head.y),
                         Point(self.head.x - (3 * BLOCK_SIZE), self.head.y),
                         Point(self.head.x - (4 * BLOCK_SIZE), self.head.y),
                         Point(self.head.x - (5 * BLOCK_SIZE), self.head.y),
                         Point(self.head.x - (6 * BLOCK_SIZE), self.head.y)]
        else:
            self.body = [self.head,
                         Point(self.head.x + BLOCK_SIZE, self.head.y),
                         Point(self.head.x + (2 * BLOCK_SIZE), self.head.y),
                         Point(self.head.x + (3 * BLOCK_SIZE), self.head.y),
                         Point(self.head.x + (4 * BLOCK_SIZE), self.head.y),
                         Point(self.head.x + (5 * BLOCK_SIZE), self.head.y),
                         Point(self.head.x + (6 * BLOCK_SIZE), self.head.y)]

    def move(self, direction):
        #clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        #idx = clock_wise.index(self.direction)
        # if np.array_equal(action, [1, 0, 0]):
        #     new_dir = clock_wise[idx]  # straight
        # elif np.array_equal(action, [0, 1, 0]):
        #     next_idx = (idx + 1) % 4
        #     new_dir = clock_wise[next_idx]  # right turn
        # else:
        #     next_idx = (idx - 1) % 4
        #     new_dir = clock_wise[next_idx]  # left turn
        # self.direction = new_dir

        x = self.head.x
        y = self.head.y
        if direction == Direction.RIGHT:
            x += BLOCK_SIZE
        elif direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif direction == Direction.DOWN:
            y += BLOCK_SIZE
        elif direction == Direction.UP:
            y -= BLOCK_SIZE
        self.head = Point(x, y)

    def is_collision(self, index, snakes):

        if len(self.body) != 0 or self.alive:
            # hits boundary
            if self.head.x > w - BLOCK_SIZE or self.head.x < 0 or self.head.y > h - BLOCK_SIZE or self.head.y < 0:
                return True

            # hits itself
            if self.head in self.body[1:]:
                return True

            # hits another snake
            for i in range(len(snakes)):
                if self.head in snakes[i].body and snakes[i] != index:
                    snakes[i].kills += 1
                    snakes[i].reward += KILL_MULT
                    return True

        return False


class SnakeGame:

    def __init__(self, snakes):
        self.w = w
        self.h = h
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('Snake')
        self.clock = pygame.time.Clock()
        self.snakes = snakes
        self.num_snakes = len(snakes)

    def play_step(self):
        # 1. collect user input
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
            if event.type == pygame.KEYDOWN:
                if self.snakes[0].alive:
                    if event.key == pygame.K_LEFT:
                        self.snakes[0].direction = Direction.LEFT
                    elif event.key == pygame.K_RIGHT:
                        self.snakes[0].direction = Direction.RIGHT
                    elif event.key == pygame.K_UP:
                        self.snakes[0].direction = Direction.UP
                    elif event.key == pygame.K_DOWN:
                        self.snakes[0].direction = Direction.DOWN
                if self.snakes[1].alive:
                    if event.key == pygame.K_a:
                        self.snakes[1].direction = Direction.LEFT
                    elif event.key == pygame.K_d:
                        self.snakes[1].direction = Direction.RIGHT
                    elif event.key == pygame.K_w:
                        self.snakes[1].direction = Direction.UP
                    elif event.key == pygame.K_s:
                        self.snakes[1].direction = Direction.DOWN
                if len(self.snakes) > 2 and self.snakes[2].alive:
                    if event.key == pygame.K_g:
                        self.snakes[2].direction = Direction.LEFT
                    elif event.key == pygame.K_j:
                        self.snakes[2].direction = Direction.RIGHT
                    elif event.key == pygame.K_y:
                        self.snakes[2].direction = Direction.UP
                    elif event.key == pygame.K_h:
                        self.snakes[2].direction = Direction.DOWN
                if len(self.snakes) > 3 and self.snakes[3].alive:
                    if event.key == pygame.K_5:
                        self.snakes[3].direction = Direction.LEFT
                    elif event.key == pygame.K_6:
                        self.snakes[3].direction = Direction.RIGHT
                    elif event.key == pygame.K_7:
                        self.snakes[3].direction = Direction.UP
                    elif event.key == pygame.K_8:
                        self.snakes[3].direction = Direction.DOWN

        # 2. move
        for snake in self.snakes:
            snake.body.insert(0, snake.head)
            snake.move(snake.direction)  # update the head
            snake.body.pop()  # update the tail

        # 3. update screen
        self._update_ui()

        # 4. check if snakes are alive
        for i in range(len(self.snakes)):
            if self.snakes[i].is_collision(i, self.snakes):
                self.snakes[i].alive = False
                self.num_snakes -= 1

        for snake in self.snakes:
            if not snake.alive and (
                    abs(snake.reward) / DEATH_MULT > len(self.snakes) or snake.reward == KILL_MULT * snake.kills):
                snake.reward -= DEATH_MULT * self.num_snakes  # minus the place it finished
                snake.body = []

        # 5. check if game is are over
        if self.num_snakes == 1:
            for snake in self.snakes:
                if snake.alive:
                    snake.reward += WIN_MULT  # adding 5 to winner
            game_over = True
        elif self.num_snakes == 0:
            game_over = True
        else:
            game_over = False

        # 6. update clock
        self.clock.tick(SPEED)

        # 7. return game over
        return game_over

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
        text = 'Player1 reward = ' + str(self.snakes[0].reward) + '  |  Player2 reward = ' + str(self.snakes[1].reward)
        if len(self.snakes) > 2:
            text += '  |  Player3 reward = ' + str(self.snakes[2].reward)
        if len(self.snakes) > 3:
            text += '  |  Player4 reward = ' + str(self.snakes[3].reward)
        self.display.blit(font.render(text, True, WHITE), [0, 0])
        pygame.display.flip()


if __name__ == '__main__':
    player1 = Snake(w / 4, h / 4, Direction.RIGHT)
    player2 = Snake(w / 4, h * 3 / 4, Direction.RIGHT)
    player3 = Snake(3 * w / 4, h / 4, Direction.LEFT)
    player4 = Snake(3 * w / 4, h * 3 / 4, Direction.LEFT)
    players = [player1, player2, player3]
    game = SnakeGame(players)

    # game loop
    while True:
        game_over = game.play_step()

        if game_over == True:
            break

    print('Player1 reward = ' + str(player1.reward) + "\nPlayer2 reward = " + str(player2.reward))
    if len(players) > 2:
        print('Player3 reward = ' + str(player3.reward))
    if len(players) > 3:
        print('Player4 reward = ' + str(player4.reward))
    pygame.quit()
