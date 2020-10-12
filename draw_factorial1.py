from matplotlib import pyplot as plt
import matplotlib.patches as mp
import math


from xugencheng.common import new_figure


W = 0.8
H = 0.8
gi = 0
g_list = []
g_false_list = []
g_colors = None
min_x = 0
max_x = 0
min_y = 0
max_y = 0


def build_colors(data_list: list):
    false_colors = [
        "#AA0000",
        "#AA0000",
        "#AA0000"]

    def i2gray(i):
        def h(num):
            return hex(num)[2:]

        gray = 256 - 32 - i * 3
        gray = h(gray)
        return "#" + gray * 3

    fi = 0
    colors = []
    for d in data_list:
        if d[1]:
            colors.append(i2gray(d[0]))
        else:
            colors.append(false_colors[fi])
            fi += 1
            assert fi <= len(false_colors)
    return colors


def draw_bin(data_list: list, cols: int, rows: int, base_x: float, base_y: float, color_list: list = None,
             sx=None, sy=None, p_true=False):
    global min_x, max_x, min_y, max_y

    if color_list is None:
        colors = build_colors(data_list)
    else:
        colors = color_list

    # 画8*8的格子
    # ax = new_figure(cols, rows, "white")
    rec_list = []
    count = 0
    all_true = True
    for d in data_list:
        if not d[1]:
            all_true = False
            break
    ex_min, ex_max, ey_min, ey_max = None, None, None, None
    for d in data_list:
        i = count // cols
        j = count % cols
        x = base_x + cols - 1 - j
        y = base_y + i
        if ex_min is None:
            ex_min = x
        if ex_max is None:
            ex_max = x
        if ey_min is None:
            ey_min = y
        if ey_max is None:
            ey_max = y

        if x < ex_min:
            ex_min = x
        if y < ey_min:
            ey_min = y
        if x > ex_max:
            ex_max = x
        if y > ey_max:
            ey_max = y

        if x < min_x:
            min_x = x
        if base_x + len(data_list) > max_x:
            max_x = base_x + len(data_list)
        if y < min_y:
            min_y = y
        if base_y + rows > max_y:
            max_y = base_y + rows

        w, h = W, H
        r = mp.Rectangle((x, y), w, h, fc=colors[d[0]], fill=True)
        # ax.add_patch(r)
        rec_list.append(r)
        count += 1

    if sx is not None and sy is not None:
        rec_list.append([sx, sy, (ex_min + ex_max + 1) / 2, ey_min, p_true])
    global g_list
    g_list.append(rec_list)
    # global gi
    # plt.savefig("fig/" + str(gi) + ".jpg")
    # gi += 1
    # plt.show()
    if len(data_list) > 1:
        data_list0 = data_list[:len(data_list) >> 1]
        data_list1 = data_list[len(data_list) >> 1:]
        i = 0
        base_y += rows + max(rows / 2, 1)
        margin = len(data_list) / 2
        base_x0 = base_x + cols / 2 - margin / 2 - len(data_list0) / 2
        base_x1 = base_x + cols / 2 + margin / 2 + len(data_list1) / 2

        while rows * cols != len(data_list0):
            if i % 2 == 0 and rows >> 1 != 0 and cols < rows * 2:
                cols = cols
                rows = rows >> 1
            else:
                cols = cols >> 1
                rows = rows
            i += 1
        assert len(data_list1) > 0
        assert len(data_list0) > 0

        base_x0 -= cols / 2
        base_x1 -= cols / 2

        draw_bin(data_list1, cols=cols, rows=rows, color_list=colors, base_x=base_x1, base_y=base_y,
                 sx=ex_max + 1, sy=(ey_min + ey_max + 1) / 2, p_true=all_true)
        draw_bin(data_list0, cols=cols, rows=rows, color_list=colors, base_x=base_x0, base_y=base_y,
                 sx=ex_min, sy=(ey_min + ey_max + 1) / 2, p_true=all_true)


def draw_factorial(data_list: list, bits: int, cols: int, rows: int, base_x: float, base_y: float,
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


def redraw_bin(file_name=None):
    ax = new_figure(max_x - min_x + 0.2, max_y - min_y + 0.2, "white")
    for recs in g_list:
        for rec in recs:
            if isinstance(rec, list):
                sx, sy, ex, ey, all_true = rec
                sx -= min_x
                ex -= min_x
                sy -= min_y
                ey -= min_y
                style = "-"
                color = "r"
                if all_true:
                    style = (0, (3, 3))
                    color = "k"
                ax.annotate("",
                            xy=(ex, ey),
                            xytext=(sx, sy),
                            arrowprops=dict(color=color,
                                            linewidth=5,
                                            arrowstyle="-|>",
                                            linestyle=style))
            else:
                rec: mp.Rectangle = rec
                rec.set_x(rec.get_x() - min_x + 0.1)
                rec.set_y(rec.get_y() - min_y + 0.1)
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
    draw_bin(data_with_label, cols=8, rows=8, color_list=g_colors, base_x=0, base_y=0)
    redraw_bin("temp.pdf")
    #
    g_list.clear()
    min_x, min_y, max_x, max_y = 0, 0, 70, 20
    import math

    # draw_factorial(data_with_label, bits=int(math.log2(64)),
    #                cols=8, rows=8, color_list=g_colors,
    #                base_x=0.1, base_y=6.1,
    #                flag=True)
    # redraw_bin("temp1.svg")
    # i = 0
    # while len(g_false_list) > 0 and i < 5:
    #     data_len = len(g_false_list)
    #     rows = int(math.sqrt(data_len))
    #     cols = data_len // rows + (1 if data_len % rows != 0 else 0)
    #     print(i)
    #     g_list.clear()
    #     draw_factorial(g_false_list, bits=int(math.log2(data_len)),
    #                    cols=cols, rows=rows, color_list=g_colors,
    #                    base_x=0.1, base_y=0.1 + rows / 2 + rows / 4,
    #                    flag=True)
    #     redraw("temp" + str(i + 5) + ".pdf")
    #     break