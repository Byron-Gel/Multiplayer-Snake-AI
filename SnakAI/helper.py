import matplotlib.pyplot as plt
from IPython import display

plt.ion()


def plot(p1, p2, p3, p4):
    display.clear_output(wait=True)
    display.display(plt.gcf())
    plt.clf()
    plt.title('Training...')
    plt.xlabel('Number of Games')
    plt.ylabel('Score')
    plt.axis([0, len(p1), 4, 1])
    plt.plot(p1, 'b')
    plt.plot(p2, 'r')
    plt.plot(p3, 'g')
    plt.plot(p4, 'y')
    plt.text(len(p1) - 1, p1[-1], str(p1[-1]))
    plt.text(len(p2) - 1, p2[-1], str(p2[-1]))
    plt.text(len(p3) - 1, p3[-1], str(p3[-1]))
    plt.text(len(p4) - 1, p4[-1], str(p4[-1]))
    plt.legend(["player1", "player2", "player3", "player4"])
    plt.show(block=False)
    plt.pause(.1)