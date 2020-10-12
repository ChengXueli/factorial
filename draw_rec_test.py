import matplotlib.pyplot as plt
import matplotlib.patches as mp

from xugencheng.common import new_figure


def main():
    # 画2*5的格子
    ax = new_figure(5, 2, "white")
    colors = ["r", "b", "y", "k", "g"]
    for i in range(2):
        for j in range(5):
            x = j
            y = i
            w, h = 1, 1
            r = mp.Rectangle((x, y), w, h,
                             fc=colors[-i + j], linewidth=1)
            ax.add_patch(r)
    plt.show()


if __name__ == '__main__':
    # main()
    a = {1, 2, 3}
    b = (3, 4, 5, 6)
    print(a.union(b))
