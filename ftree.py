import random
import sys
import math
from common import *
from draw_factorial2 import build_colors
import matplotlib.patches as mp
from matplotlib import pyplot as plt

REPEAT = 1

PROB = 2
PROB_BASE = 64
# 元素标签长度
LEN = PROB_BASE

W = 0.8
H = 0.8
gi = 0
g_list = []
g_false_list = []
g_txt = []
g_colors = None
g_min_x = 0
g_max_x = 0
g_min_y = 0
g_max_y = 0

f_min_x = 0
f_max_x = 0
f_min_y = 0
f_max_y = 0


#
# 给定一个数及其比特数，返回移除第nth 二进制位的后的数
#
# def trim_bits(number, bits, nth):
#     assert (number >> bits) == 0
#     assert bits > nth
#     assert nth >= 0
#
#     if nth == 0:
#         return number >> 1
#
#     high = number >> (nth + 1)
#     low = ((1 << nth) - 1) & number
#
#     return (high << nth) | low
# def reverse_bits(number, bits):
#     mask = (1 << bits) - 1
#     return mask ^ number
def get_rows_cols(length):
    rows = int(math.sqrt(length))
    cols = length // rows + (1 if length % rows != 0 else 0)
    return rows, cols


#
# 获取第nth 二进制位为1的数的集合
#
def get_special_set(objects, bits, nth, set):
    assert bits > nth
    v = list()
    for obj in objects:
        if ((1 << nth) & num(obj)) == (set << nth):
            v.append([num(obj), label(obj), valid(obj), True])
        else:
            v.append([num(obj), label(obj), valid(obj), False])
    return v


def verify(p):
    return valid(p)


#
# 返回聚合验证的结果, 如果全部验证通过，返回True，否则返回False
#
def aggregate_verify(objects, start=0, end=-1):
    if end < start:
        end = len(objects)
    for it in objects[start:end]:
        if selected(it) and not verify(it):
            return False
    return True


def remove_dup(a_list: list):
    d = {label(data): data for data in a_list}
    return list(d.values())


def draw(objects, cols, rows, base_x, base_y, true_set=None, colors=None):
    assert cols * rows >= len(objects)
    global g_min_x, g_min_y, g_max_x, g_max_y
    rec_list = []
    if colors is None:
        colors = build_colors(objects, true_set)
        assert len(colors) == len(objects)
    for count, obj in enumerate(objects):
        i = count // cols
        j = count % cols
        x = base_x + cols - 1 - j
        y = base_y + i

        w, h = W, H
        r = mp.Rectangle((x, y), w, h, fc=colors[num(obj)], fill=selected(obj))
        rec_list.append(r)

        if g_min_x == g_max_y == g_min_y == g_max_x == 0:
            g_min_x = x
            g_max_x = x + 1
            g_min_y = y
            g_max_y = y + 1

        if x < g_min_x:
            g_min_x = x
        if x + 1 > g_max_x:
            g_max_x = x + 1

        if y < g_min_y:
            g_min_y = y
        if y + 1 > g_max_y:
            g_max_y = y + 1
    global g_list
    g_list.append(rec_list)
    return colors


class MyArrow(mp.FancyArrow):
    def __init__(self, x, y, dx, dy, **kwargs):
        self.x = x
        self.y = y
        self.dx = dx
        self.dy = dy
        self.kwargs = kwargs
        super().__init__(x, y, dx, dy, **kwargs)

    def get_x(self):
        return self.x

    def get_y(self):
        return self.y

    def set_x(self, x):
        self.x = x
        self.__init__(self.x, self.y, self.dx, self.dy, **self.kwargs)

    def set_y(self, y):
        self.y = y
        self.__init__(self.x, self.y, self.dx, self.dy, **self.kwargs)


def redraw(file_name=None, random_arr=None, num_fakes=None):
    sp = 0.1
    ax = new_figure(g_max_x - g_min_x + sp * 2, g_max_y - g_min_y + sp * 2, "white")

    for x, y, s in g_txt:
        plt.text(x - g_min_x + sp, y - g_min_y + sp, s, size=50)
    for recs in g_list:
        for rec in recs:
            if isinstance(rec, mp.Rectangle):
                rec: mp.Rectangle = rec
                rec.set_x(rec.get_x() - g_min_x + sp)
                rec.set_y(rec.get_y() - g_min_y + sp)
                ax.add_patch(rec)
            elif isinstance(rec, MyArrow):
                rec: MyArrow = rec
                rec.set_x(rec.get_x() - g_min_x + sp)
                rec.set_y(rec.get_y() - g_min_y + sp)
                ax.add_patch(rec)

    if file_name is not None:
        plt.savefig(str(file_name))
    # plt.show()


# 阶乘树划分
def find_fake_factorial(objects, bits,
                        cols: int, rows: int,
                        base_x: float, base_y: float,
                        r2l=False, single_line=True, vertical=False,
                        padding_for_filter_inner=1, margin_for_filter=2):
    """

    :param objects:
    :param bits: 位数
    :param cols: 要画图的列数
    :param rows: 要画图的行数
    :param base_x: 当前图的x坐标
    :param base_y: 当前图的y坐标
    :param r2l: 图像是否从右往左
    :param single_line:
    :param vertical:
    :param padding_for_filter_inner:
    :param margin_for_filter:
    :return:
    """

    # c_max_x, c_min_x, c_max_y, c_min_y = 0, 0, 0, 0

    result = list()
    local_seq_count = 0
    local_par_count = 0

    all_true = aggregate_verify(objects)

    # 当前filter的中心线
    center_x = base_x + cols / 2
    center_y = base_y + rows / 2

    colors = draw(objects, cols, rows, base_x, base_y)

    assert len(objects) != 0
    if len(objects) == 0:
        return result, local_seq_count, local_par_count

    if len(objects) == 1:
        if not aggregate_verify(objects):
            result.append(objects[0])
        return result, local_seq_count, local_par_count

    # 此列表存放的一定含有假，有多少不清楚。
    list_list_false = list()

    # 此集合存放的一定为真, 只存放了label,即原始编号
    true_set_labels = set()

    sign = -1 if r2l else 1
    if vertical:
        if single_line:
            base_x0 = base_x
            base_x1 = base_x0
            base_y0 = base_y + (rows + margin_for_filter)
            base_y1 = base_y0 + (rows + padding_for_filter_inner)
        else:
            # TODO fix
            base_x0 = base_x + (cols + margin_for_filter)
            base_x1 = base_x0
            base_y0 = center_y - padding_for_filter_inner / 2 - rows
            base_y1 = center_y + padding_for_filter_inner / 2
    else:
        if single_line:
            base_y0 = base_y
            base_y1 = base_y0
            base_x0 = base_x + (cols + margin_for_filter) * sign
            base_x1 = base_x0 + (cols + padding_for_filter_inner) * sign
        else:
            base_y0 = center_y - padding_for_filter_inner / 2 - rows
            base_y1 = center_y + padding_for_filter_inner / 2
            base_x0 = base_x + (cols + margin_for_filter) * sign
            base_x1 = base_x0

    # for outline
    padding_for_outline = 0.5

    # x, y 分别为当前filter中各个分组的x, y的最小值，再减去轮廓的padding
    x, y, w, h = base_x0 - padding_for_outline, base_y0 - padding_for_outline, 0, 0
    if vertical:
        if single_line:
            w = cols + 2 * padding_for_outline
        else:
            w = 2 * cols + padding_for_filter_inner + 2 * padding_for_outline
    else:
        if single_line:
            h = rows + 2 * padding_for_outline
        else:
            h = 2 * rows + padding_for_filter_inner + 2 * padding_for_outline

    # 存放每一个分组的数据，及在图形中的x,y坐标，每个元素形如：[data, x, y]
    global g_list, g_txt
    global g_min_x, g_min_y, g_max_x, g_max_y

    # 进行filter过程，包括画图
    for i in range(bits):
        p0 = get_special_set(objects, bits, i, 0)
        p1 = get_special_set(objects, bits, i, 1)

        draw(p0, cols, rows, base_x0, base_y0)
        draw(p1, cols, rows, base_x1, base_y1)

        if base_y0 - 1 < g_min_y:
            g_min_y = base_y0 - 1
        text0 = "$S_{%d}^{0}$" % (i + 1)
        text1 = "$S_{%d}^{1}$" % (i + 1)
        if vertical:
            g_txt.append([base_x0 + cols + 1, base_y0 + rows / 2, text0])
            if single_line:
                g_txt.append([base_x1 + cols + 1, base_y1 + rows / 2, text1])
            else:
                g_txt.append([base_x1 + cols + 1, base_y1 + rows / 2, text1])
        else:
            g_txt.append([base_x0 + 1, base_y0 - 1, text0])
            if single_line:
                g_txt.append([base_x1 + 1, base_y1 - 1, text1])
            else:
                g_txt.append([base_x1 + 1, base_y1 + rows + 2, text1])

        ls = (0, (20, 20))
        color = 'r'
        margin = 0.2
        if vertical:
            g_list.append(
                [mp.Rectangle((base_x0 - margin, base_y0 - margin),
                              cols + 2 * margin,
                              rows * 2 + padding_for_filter_inner + 2 * margin,
                              edgecolor=color, fill=False, linestyle=ls)])
        else:
            if r2l:
                if single_line:
                    g_list.append(
                        [mp.Rectangle((base_x1 - margin, base_y1 - margin),
                                      2 * cols + padding_for_filter_inner + margin * 2,
                                      rows + margin * 2,
                                      edgecolor=color, fill=False, linestyle=ls)])
                else:
                    g_list.append(
                        [mp.Rectangle((base_x1 - margin, base_y1 - margin),
                                      cols + margin * 2,
                                      2 * rows + padding_for_filter_inner + margin * 2,
                                      edgecolor=color, fill=False, linestyle=ls)])
            else:
                if single_line:
                    g_list.append(
                        [mp.Rectangle((base_x0 - margin, base_y0 - margin),
                                      2 * cols + padding_for_filter_inner + margin * 2,
                                      rows + margin * 2,
                                      edgecolor=color, fill=False, linestyle=ls)])
                else:
                    g_list.append(
                        [mp.Rectangle((base_x0 - margin, base_y0 - margin),
                                      cols + margin * 2,
                                      2 * rows + padding_for_filter_inner + margin * 2,
                                      edgecolor=color, fill=False, linestyle=ls)])
        x = min(x, min(base_x0, base_x1) - padding_for_outline)
        y = min(y, min(base_y0, base_y1) - padding_for_outline)

        if len(p0) != 0:
            if aggregate_verify(p0):
                true_set_labels = true_set_labels.union({label(it) for it in p0 if selected(it)})
            else:
                list_list_false.append((list(filter(selected, p0)), base_x0 + 1, base_y0 + rows + 5))

        if len(p1) != 0:
            if aggregate_verify(p1):
                true_set_labels = true_set_labels.union({label(it) for it in p1 if selected(it)})
            else:
                list_list_false.append((list(filter(selected, p1)), base_x1 + 1, base_y1 + rows + 5))
        # prepare for next looper
        if vertical:
            h += rows + padding_for_filter_inner

            base_y0 = base_y1 + (rows + padding_for_filter_inner)
            if single_line:
                base_y1 = base_y0 + (rows + padding_for_filter_inner)
                h += cols + padding_for_filter_inner
            else:
                base_x1 = base_x0
        else:
            w += cols + padding_for_filter_inner

            base_x0 = base_x1 + (cols + padding_for_filter_inner) * sign
            if single_line:
                base_x1 = base_x0 + (cols + padding_for_filter_inner) * sign
                w += cols + padding_for_filter_inner
            else:
                base_x1 = base_x0

    if vertical:
        h -= padding_for_filter_inner
        h += padding_for_outline * 2
    else:
        w -= padding_for_filter_inner
        w += padding_for_outline * 2

    # 画filter的外框

    if x < g_min_x:
        g_min_x = x
    if y < g_min_y:
        g_min_y = y
    if x + w > g_max_x:
        g_max_x = x + w
    if y + h > g_max_y:
        g_max_y = y + h
    g_list.append(
        [mp.Rectangle((x, y), w, h, fc='k', fill=False)])
    if vertical:
        base_x = center_x - cols / 2
        base_y = y + h - padding_for_outline
    else:
        if r2l:
            base_x = x - cols + padding_for_outline
        else:
            base_x = x + w - padding_for_outline
        base_y = center_y - rows / 2

    global f_min_x, f_min_y, f_max_x, f_max_y
    if f_min_x is None:
        f_min_x = x + padding_for_outline
        f_min_y = y + padding_for_outline
        f_max_x = x + w - padding_for_outline
        f_max_y = y + h - padding_for_outline

    # 将filter后的原始图像画出，确定为真的地方标注为蓝色
    if vertical:
        base_y += margin_for_filter
    else:
        base_x += margin_for_filter * sign

    draw(objects, cols, rows, base_x, base_y, true_set_labels)
    reduce = len(true_set_labels) > 0

    # 说明此次过滤起了作用，那么继续进行递归，需要重编号
    # 画出缩减后的可能为假的列表
    if reduce:
        if vertical:
            base_y += (rows + 1)
        else:
            base_x += (cols + 1) * sign

        temp = list()
        for list_false, _, _ in list_list_false:
            for it in list_false:
                # 过滤掉不可能为假的标签
                if label(it) not in true_set_labels:
                    temp.append([num(it), label(it), valid(it), True])

        list_list_false.clear()
        if len(temp) > 0:
            temp = remove_dup(temp)
            # 重编号
            temp = sorted(temp, key=lambda data: label(data))

            data_len = len(temp)
            rows, cols = get_rows_cols(data_len)
            if vertical:
                base_x = center_x - cols / 2
            else:
                base_y = center_y - rows / 2
            draw(temp, cols, rows, base_x, base_y, colors=colors)

            wi = 3
            if len(temp) > 1:
                g_list.append([
                    MyArrow(base_x + cols / 2, base_y + rows + 2, 0, 5, head_width=wi, head_length=3, width=1)
                ])
                g_txt.append([base_x + cols / 2 + 1, base_y + rows + 5, "Reorder"])
            if base_x + cols / 2 + wi / 2 > g_max_x:
                g_max_x = base_x + cols / 2 + wi / 2
            temp = [[i, label(t), valid(t), True] for i, t in enumerate(temp)]
            list_list_false.append((temp, base_x, g_max_y + 5))
            bits = int(math.log2(data_len))
            if (1 << bits) < data_len:
                bits += 1
    else:
        bits = bits - 1

    for list_false, base_x, base_y in list_list_false:
        # 如果可能为假的集合只剩一个元素，那么该元素一定为假
        if len(list_false) == 1:
            result.extend(list_false)
        else:
            list_false = [[i, label(data), valid(data), True] for i, data in enumerate(list_false)]
            rows, cols = get_rows_cols(len(list_false))
            if reduce:
                margin_for_filter *= 5
                padding_for_filter_inner *= 5
                if not r2l:
                    list_false, _, _ = find_fake_factorial(list_false, bits, cols, rows, base_x, base_y,
                                                           r2l=not r2l, vertical=vertical,
                                                           margin_for_filter=margin_for_filter,
                                                           padding_for_filter_inner=padding_for_filter_inner)
                else:
                    list_false, _, _ = find_fake_factorial(list_false, bits, cols, rows, base_x, base_y,
                                                           r2l=not r2l, vertical=vertical,
                                                           margin_for_filter=margin_for_filter,
                                                           padding_for_filter_inner=padding_for_filter_inner)
            else:
                g_list.append([
                    MyArrow(base_x + cols / 2, base_y - 4, 0, 2, head_width=1.5, head_length=1, width=0.5)
                ])
                g_txt.append([base_x + cols / 2 + 1, base_y - 2.5, "Reorder"])
                list_false, _, _ = find_fake_factorial(list_false, bits, cols, rows, base_x, base_y, vertical=True)
            result.extend(list_false)

    return sorted(remove_dup(result), key=lambda data: label(data)), local_seq_count, local_par_count


# **
# 二叉树划分
# */
def find_fake_binary(objects, start, end, do_agg_verify):
    result = list()
    local_seq_count = 0
    local_par_count = 0

    if len(objects) == 0 or start > end:
        return result, local_seq_count, local_par_count

    # 聚合验证通过，直接返回空集合

    if do_agg_verify:
        success = aggregate_verify(objects, start, end + 1)
        local_seq_count += 1
        local_par_count += 1
        if success:
            return result, local_seq_count, local_par_count

    # 如果只有一个元素，并且运行到这儿，肯定说明该元素为False
    if start == end:
        result.append(objects[start])
        return result, local_seq_count, local_par_count

    mid = (start + end + 1) >> 1

    # 前半部分
    v, seq_count, par_count0 = find_fake_binary(objects, start, mid - 1, True)
    result.extend(v)
    local_seq_count += seq_count

    # 后半部分
    v, seq_count, par_count1 = find_fake_binary(objects, mid, end, len(result) != 0)
    par_count1 += 1 if len(result) == 0 else 0
    result.extend(v)
    local_seq_count += seq_count
    local_par_count += max(par_count0, par_count1)
    return result, local_seq_count, local_par_count


def print_status(objects):
    for it in objects:
        assert (not verify(it))
        print(label(it), end=" ")
    print()


def validation(fakes, num_fakes):
    # validation
    consistency = num_fakes == len(fakes)
    if consistency:
        for p in fakes:
            if valid(p):
                return False
    return consistency


def main(argc, argv):
    global PROB, PROB_BASE, LEN, REPEAT
    global g_list, g_txt
    if argc == 2:
        PROB = int(argv[1])
        PROB_BASE = 100
        LEN = PROB_BASE
    elif argc == 3:
        PROB = int(argv[1])
        PROB_BASE = int(argv[2])
        LEN = PROB_BASE
    elif argc == 4:
        PROB = int(argv[1])
        PROB_BASE = int(argv[2])
        LEN = int(argv[3])
    elif argc == 5:
        PROB = int(argv[1])
        PROB_BASE = int(argv[2])
        LEN = int(argv[3])
        REPEAT = int(argv[4])
    b_count = 0
    f_count = 0
    s_b_count = 0
    s_f_count = 0

    random_arr = list()
    objects = list()

    num_fakes = LEN * PROB // PROB_BASE

    bits = 0
    lg = LEN - 1
    while lg != 0:
        bits += 1
        lg >>= 1

    # random.seed(0)
    print("#fakes:", num_fakes, end="\n\n")

    do_log = REPEAT <= 10
    for times in range(REPEAT):

        random_arr.clear()
        objects.clear()

        # pair: the first one is the value, the second one is always the orginal index.
        for i in range(LEN):
            objects.append([i, i, True, True])
            random_arr.append(i)

        # 采用fisher算法，产生随机序列
        i = LEN - 1
        while i != 0:
            j = random.randint(0, i)
            temp = random_arr[i]
            random_arr[i] = random_arr[j]
            random_arr[j] = temp
            i -= 1
        # data[10] = False  # 001 010 -> 10
        # data[35] = False  # 100 101 -> 35
        # random_arr[0] = 10
        # random_arr[1] = 35
        for i in range(num_fakes):
            objects[random_arr[i]][2] = False
        #     pass

        if do_log:
            for i, obj in enumerate(objects):
                print(1 if valid(obj) else 0, end=" ")
                if (i + 1) % 20 == 0:
                    print()
            print()
        fakes0, seq, par = find_fake_binary(objects, 0, len(objects) - 1, True)
        if do_log:
            print("bin:", end="\t")
            print_status(fakes0)
            print("steps:", seq, par, end="\t")
        if validation(fakes0, num_fakes):
            if do_log:
                print("OK")
        else:
            print("ER")
        global g_list

        fakes1, seq, par = find_fake_factorial(objects, bits, 8, 8, 0, 0, single_line=False)

        y = 10

        def get_bin(number):
            b = bin(number)[2:]
            return "0" * (6 - len(b)) + b

        for number in random_arr[:num_fakes]:
            g_txt.append([2.5, y + 0.8, get_bin(number)])
            g_list.append([
                mp.Rectangle((1, y), 1, 1, fc="#AA0000", fill=True)
            ])
            y += 2

        redraw("temp.pdf", random_arr, num_fakes)
        if do_log:
            print("fac:", end="\t")
            print_status(fakes1)
            print("steps:", seq, par, end="\t")
        if validation(fakes1, num_fakes):
            if do_log:
                print("OK")
        else:
            print("ER")
        print()
    print("           |   seq \t\t  par")
    print("-----------+-------------------")
    print("binary     |  ", b_count // REPEAT, " \t\t ", s_b_count // REPEAT)
    print("factorial  |  ", f_count // REPEAT, " \t\t ", s_f_count // REPEAT)


if __name__ == '__main__':
    main(len(sys.argv), sys.argv)
