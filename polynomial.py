# -*- coding: utf-8 -*-
import tensorflow as tf
import xlrd
import numpy as np
import matplotlib.pyplot as plt


def xl_read(file_path, sheet_name, title=True):
    """
    read excel file
    :param file_path:
    :param sheet_name:
    :param title: is contained title
    :return: data list
    """
    data = xlrd.open_workbook(file_path)
    table = data.sheet_by_name(sheet_name)
    nrows = table.nrows
    data = []
    for i in range(nrows):
        if title and i == 0:
            continue
        data.append(table.row_values(i))
    return data


def show_plt(x, y):
    """
    show scatter plot
    :param x:
    :param y:
    :return:
    """
    fig, ax = plt.subplots()
    ax.scatter(x, y, s=2, color="red", label="L-V")
    plt.show()


def show_plt_poly(x, y, W_test):
    """
    show scatter plot and fitting polynomial
    :param x:
    :param y:
    :param W_test: multinomial coefficient
    :return:
    """
    fig, ax = plt.subplots()
    ax.scatter(x, y, s=2, color="red", label="L-V")
    y_pred = None
    for i in range(len(W_test)):
        if i == 0:
            y_pred = W_test[i]
        else:
            y_pred = y_pred + np.power(x, i) * W_test[i]
    ax.plot(x, y_pred)
    plt.show()


if __name__ == '__main__':
    excel_file = r"data.xlsx"
    data_list = xl_read(excel_file, 'Sheet1')
    data_array = np.array(data_list)
    # data L-V
    level = data_array[:, 2]
    volume = data_array[:, 1]
    show_plt(level, volume)

    # 定义容器,放入X和Y
    X = tf.placeholder(tf.float32, name="X")
    Y = tf.placeholder(tf.float32, name="Y")
    # 定义权重和偏置,对应与一次线性函数y = wx + b

    W_list = []
    Y_pred = None
    n = 2  # 多项式的最高次幂
    learning_rate = 0.000028  # 二分法确定梯度下降步长
    # n = 2; rate = 0.00028
    # n = 3; rate = 0.0000000087
    for i in range(n + 1):
        W = tf.Variable(tf.random_normal([1]), name="weight_" + str(i))
        W_list.append(W)
        if i == 0:
            Y_pred = W
        else:
            Y_pred = tf.add(tf.multiply(tf.pow(X, i), W), Y_pred)

    sample_num = level.shape[0]  # 取出xs的个数，这里是100个
    loss = tf.reduce_sum(tf.pow(Y_pred - Y, 2)) / sample_num  # 向量对应的点相减之后，求平方和，在除以点的个数

    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

    n_samples = level.shape[0]
    print(n_samples)
    init = tf.global_variables_initializer()
    with tf.Session() as sess:

        # 初始化所有变量
        sess.run(init)  # 将搜集的变量写入事件文件，提供给Tensorboard使用
        # writer = tf.summary.FileWriter('./graphs/polynomial_reg', sess.graph)
        # http://localhost:6006/
        # cmd
        # tensorboard --logdir=./graphs/polynomial_reg
        # 训练模型
        for i in range(1000):
            total_loss = 0  # 设定总共的损失初始值为0
            for x, y in zip(level, volume):  # zip:将两个列表中的对应元素分别取一个出来，形成一个元组 _,
                _, l = sess.run([optimizer, loss], feed_dict={X: x, Y: y})
                # print(l)
                total_loss += l  # 计算所有的损失值进行叠加
            if i % 100 == 0:
                print('Epoch {0}: {1}'.format(i, total_loss / n_samples))  # 关闭
                W_test = sess.run(W_list)
                print(W_test)
                show_plt_poly(level, volume, W_test)

        #writer.close()  # 取出w和b的值
        W_result = sess.run(W_list)

    print(W_result)
