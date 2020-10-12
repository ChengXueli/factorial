import matplotlib.pyplot as plt


def num(pi):
    return pi[0]


def label(pi):
    return pi[1]


def valid(pi):
    return pi[2]


def selected(pi):
    return pi[3]


def new_figure(w, h, color):
    plt.figure(figsize=(w, h))
    ax = plt.gca()
    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    ax.xaxis.tick_top()
    ax.invert_yaxis()
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['top'].set_color(color)
    ax.spines['right'].set_color(color)
    ax.spines['bottom'].set_color(color)
    ax.spines['left'].set_color(color)

    ax.spines['bottom'].set_linewidth(1)
    ax.spines['left'].set_linewidth(1)
    ax.spines['top'].set_linewidth(1)
    ax.spines['right'].set_linewidth(1)
    return ax
