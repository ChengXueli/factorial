import random
import sys
import math
from common import *
from draw_factorial2 import build_colors
import matplotlib.patches as mp
from matplotlib import pyplot as plt

REPEAT = 1

PROB = 5
PROB_BASE = 20
# 元素标签长度
LEN = PROB_BASE

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


def draw(objects, cols, rows, base_x, base_y, true_set=None):
    assert cols * rows >= len(objects)
    global min_x, min_y, max_x, max_y
    rec_list = []
    colors = build_colors(objects, true_set)
    for count, obj in enumerate(objects):
        i = count // cols
        j = count % cols
        x = base_x + cols - 1 - j
        y = base_y + i

        w, h = W, H
        r = mp.Rectangle((x, y), w, h, fc=colors[num(obj)], fill=selected(obj))
        rec_list.append(r)

        if min_x == max_y == min_y == max_x == 0:
            min_x = x
            max_x = x + 1
            min_y = y
            max_y = y + 1

        if x < min_x:
            min_x = x
        if x + 1 > max_x:
            max_x = x + 1

        if y < min_y:
            min_y = y
        if y + 1 > max_y:
            max_y = y + 1
    global g_list
    g_list.append(rec_list)


def redraw(file_name=None):
    sp = 0.01
    ax = new_figure(max_x - min_x + sp * 2, max_y - min_y + sp * 2, "white")
    for recs in g_list:
        for rec in recs:
            if isinstance(rec, mp.Rectangle):
                rec: mp.Rectangle = rec
                rec.set_x(rec.get_x() - min_x + sp)
                rec.set_y(rec.get_y() - min_y + sp)
                ax.add_patch(rec)
    if file_name is not None:
        plt.savefig(str(file_name))
    plt.show()


# 阶乘树划分
def find_fake_factorial(objects, bits,
                        cols: int, rows: int,
                        base_x: float, base_y: float,
                        r2l=False):
    """

    :param objects:
    :param bits: 位数
    :param cols: 要画图的列数
    :param rows: 要画图的行数
    :param base_x: 当前图的x坐标
    :param base_y: 当前图的y坐标
    :param r2l: 图像是否从右往左
    :return:
    """
    result = list()
    local_seq_count = 0
    local_par_count = 0

    all_true = aggregate_verify(objects)
    center_x = base_x + cols / 2
    center_y = base_y + rows / 2

    draw(objects, cols, rows, base_x, base_y)

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

    base_x += cols + 2
    base_y0 = base_y - rows - 0.5
    base_y1 = base_y0 + 0.5

    # 进行filter过程，包括画图
    for i in range(bits):
        s0 = get_special_set(objects, bits, i, 0)
        s1 = get_special_set(objects, bits, i, 1)

        draw(s0, cols, rows, base_x, base_y0)
        draw(s1, cols, rows, base_x, base_y1)

        base_x += cols + 1

        if len(s0) != 0:
            if aggregate_verify(s0):
                true_set_labels = true_set_labels.union({label(it) for it in s0 if selected(it)})
            else:
                list_list_false.append(list(filter(selected, s0)))

        if len(s1) != 0:
            if aggregate_verify(s1):
                true_set_labels = true_set_labels.union({label(it) for it in s1 if selected(it)})
            else:
                list_list_false.append(list(filter(selected, s1)))

    # 将filter后的原始图像画出，确定为真的地方标注为蓝色
    base_x += 1
    draw(objects, cols, rows, base_x, base_y, true_set_labels)
    reduce = len(true_set_labels) > 0

    # 说明此次过滤起了作用，那么继续进行递归，需要重编号
    # 画出缩减后的可能为假的列表
    if reduce:
        temp = list()
        new_order = 0
        for list_false in list_list_false:
            for it in list_false:
                # 过滤掉不可能为假的标签
                if label(it) not in true_set_labels:
                    temp.append([new_order, label(it), valid(it)])
                    new_order += 1

        list_list_false.clear()
        if len(temp) > 0:
            temp = remove_dup(temp)
            data_len = len(temp)
            rows = int(math.sqrt(data_len))
            cols = data_len // rows + (1 if data_len % rows != 0 else 0)

            base_x += cols + 1
            base_y = center_y - rows / 2
            draw(temp, cols, rows, base_x, base_y)

            list_list_false.append(temp)
            bits = int(math.log2(len(temp)))
            if (1 << bits) < len(temp):
                bits += 1
    else:
        bits = bits - 1

    for list_false in list_list_false:
        # 如果可能为假的集合只剩一个元素，那么该元素一定为假
        if len(list_false) == 1:
            result.extend(list_false)
        else:

            list_false, _, _ = find_fake_factorial(list_false, bits)
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

        for i in range(num_fakes):
            objects[random_arr[i]][2] = False
            pass
        # data[10] = False  # 001 010 -> 10
        # data[14] = False  # 001 110 -> 14
        # data[35] = False  # 100 101 -> 35
        # objects[2][2] = False
        # objects[14][2] = False
        # objects[35][2] = False
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

        fakes1, seq, par = find_fake_factorial(objects, bits)
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
