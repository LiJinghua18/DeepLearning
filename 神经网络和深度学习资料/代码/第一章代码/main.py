
import network
import mnist_loader

def main():
    training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

    """该网络有三层:
       第一层：784个神经元（图像由多少维表示就有多少输入神经元，手写数字识别的图片都是784=28*28的）
       第二层：30个神经元
       第三层：10个神经元（最后需要分几类就设置多大个神经元）
       """
    net = network.Network([784, 30, 10])
    """
    SGD的参数分别为：训练数据、迭代次数、数据一次加载量、学习率、测试数据
    """
    net.SGD(training_data, 30, 10, 3.0, test_data=test_data)


if __name__ == '__main__':
    main()
