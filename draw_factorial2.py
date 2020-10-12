import math
import copy
import matplotlib.patches as mp
from matplotlib import pyplot as plt

from common import *

W = 0.8
H = 0.8
gi = 0
g_list = []
g_false_list = []
g_colors = None
g_min_x = 0
g_max_x = 0
g_min_y = 0
g_max_y = 0


def build_colors(data_list: list, true_set=None):
    global g_colors

    SC = 32
    EC = 224
    step = (EC - SC) // len(data_list)
    false_colors = [
        "#CC0000",
        "#CC00CC",
        "#AA0000"]
    true_colors = [
        "#40A0FF",
        "#40A0FF",
        "#40A0FF",
    ]

    def i2gray(i):
        def h(number):
            return hex(number)[2:]

        gray = EC - i * step - step // 2
        gray = h(gray)
        return "#" + gray * 3

    fi = 0
    colors = []
    for d in data_list:
        if true_set is not None and label(d) in true_set:
            colors.append(true_colors[0])
        elif valid(d):
            colors.append(i2gray(num(d)))
        else:
            if g_colors is not None:
                colors.append(g_colors[label(d)])
            else:
                colors.append(false_colors[fi % len(false_colors)])
            fi += 1
    if g_colors is None:
        g_colors = colors
    return colors


def draw_factorial(data_list: list, bits: int,
                   cols: int, rows: int,
                   base_x: float, base_y: float,
                   color_list: list = None, flag=False):
    if color_list is None:
        colors = build_colors(data_list)
    else:
        colors = color_list
    global g_colors
    g_colors = colors
    # 画8*8的格子
    # ax = new_figure(cols, rows, "white")
    rec_list = []
    count = 0
    all_true = True
    for d in data_list:
        if not d[1] and d[2]:
            all_true = False
            break
    for d in data_list:
        i = count // cols
        j = count % cols
        x = base_x + cols - 1 - j
        y = base_y + i

        w, h = W, H
        r = mp.Rectangle((x, y), w, h, fc=colors[d[0]], fill=d[2])
        # ax.add_patch(r)
        rec_list.append(r)
        count += 1
    global g_list
    g_list.append(rec_list)

    if flag:
        def data_filter(data_list: list, nth: int, zero_or_one: int):
            result = list()
            for d in data_list:
                if (d[0] & (1 << nth)) == (zero_or_one << nth):
                    result.append([d[0], d[1], True])
                else:
                    result.append([d[0], d[1], False])
            return result

        base_x += cols + 2
        base_y0 = 0.1
        base_y1 = base_y0 + rows + rows / 2
        all_true_list = dict()
        false_list = dict()

        for nth in range(bits):
            data_list0 = data_filter(data_list, nth, 0)
            data_list1 = data_filter(data_list, nth, 1)
            i = 0
            i += 1
            if len(data_list1) > 0:
                all_true = draw_factorial(data_list1, bits=bits, cols=cols, rows=rows, color_list=colors,
                                          base_x=base_x, base_y=base_y1, flag=False)
                if all_true:
                    for d in data_list1:
                        if d[2]:
                            all_true_list[d[0]] = d
                else:
                    for d in data_list1:
                        if d[2]:
                            false_list[d[0]] = d

            if len(data_list0) > 0:
                all_true = draw_factorial(data_list0, bits=bits, cols=cols, rows=rows, color_list=colors,
                                          base_x=base_x, base_y=base_y0, flag=False)
                if all_true:
                    for d in data_list0:
                        if d[2]:
                            all_true_list[d[0]] = d
                else:
                    for d in data_list0:
                        if d[2]:
                            false_list[d[0]] = d

            base_x += cols + 1
            base_y = (base_y0 + rows + base_y1) / 2

        final_false_list = []
        for key, value in false_list.items():
            if all_true_list.get(key) is None:
                final_false_list.append(value)
        temp_list = []

        data_len = len(final_false_list)
        if data_len != 0:
            rows = int(math.sqrt(data_len))
            cols = data_len // rows + (1 if data_len % rows != 0 else 0)

            base_x += 1
            base_y -= rows / 2
            global g_false_list
            g_false_list = sorted(final_false_list, key=lambda d: d[0])
            draw_factorial(g_false_list, bits=bits, cols=cols, rows=rows, color_list=colors, base_x=base_x,
                           base_y=base_y, flag=False)
    return all_true


def redraw(file_name=None):
    ax = new_figure(g_max_x - g_min_x + 0.2, g_max_y - g_min_y + 0.2, "white")
    for recs in g_list:
        for rec in recs:
            rec: mp.Rectangle = rec
            rec.set_x(rec.get_x() - g_min_x + 0.1)
            rec.set_y(rec.get_y() - g_min_y + 0.1)
            ax.add_patch(rec)
    if file_name is not None:
        plt.savefig(str(file_name))
    plt.show()


if __name__ == '__main__':
    data = [True] * 64
    data[10] = False  # 001 010 -> 10
    data[14] = False  # 001 110 -> 14
    data[35] = False  # 100 101 -> 35
    data_with_label = [[i, d, True] for i, d in enumerate(data)]

    g_min_x, g_min_y, g_max_x, g_max_y = 0, 0, 70, 50

    # draw_factorial(data_with_label, bits=int(math.log2(len(data_with_label))),
    #                cols=8, rows=8, color_list=g_colors,
    #                base_x=0.1, base_y=6.1,
    #                flag=True)
    pre = []
    g_false_list = data_with_label
    i = 0
    while len(g_false_list) != len(pre):
        data_with_label = copy.deepcopy(g_false_list)
        g_false_list.clear()

        data_len = len(data_with_label)
        rows = int(math.sqrt(data_len))
        cols = data_len // rows + (1 if data_len % rows != 0 else 0)
        g_list.clear()
        draw_factorial(data_with_label, bits=int(math.log2(data_len)),
                       cols=cols, rows=rows, color_list=g_colors,
                       base_x=0.1, base_y=0.1 + rows / 2 + rows / 4,
                       flag=True)
        print(i)
        i += 1
        if i == 2:
            break
    redraw("temp" + str(i + 5) + ".pdf")
