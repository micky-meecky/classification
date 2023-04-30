# 写一个tensorboard的学习案例
# 1.导入包

import tensorboardX

if __name__ == '__main__':
    # 2.定义一个SummaryWriter对象
    # writer = tensorboardX.SummaryWriter('log')
    # for()模拟训练过程
    epoch_num = 100
    loss = 1.0
    writer = tensorboardX.SummaryWriter('log')
    for i in range(epoch_num):
        # 模拟每个epoch中的batch训练过程
        for j in range(100):
            loss *= 0.99
        # 3.写入训练过程中的loss值
        writer.add_scalar('loss', loss, i)
    # 4.关闭writer对象
    writer.close()
