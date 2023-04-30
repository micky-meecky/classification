import datetime
import time


class TicToc:
    def __init__(self):
        # tic 和 toc 是两个 datetime.datetime 对象，是用来计算时间差的，toc_list 是一个列表，用来存储多次计算的时间差
        self.tic = 0
        self.toc = 0
        self.toc_list = []
        self.total_seconds = 0
        self.hours = 0
        self.minutes = 0
        self.seconds = 0
        self.single_seconds = 0
        self.wholetime = 0
        self.wholehours = 0
        self.wholemintues = 0
        self.wholeseconds = 0

    def ticbegin(self):
        self.tic = datetime.datetime.now()

    def ticend(self):
        self.toc = datetime.datetime.now()
        if self.tic == 0:   # 如果没有开始时间，就抛出异常
            self.tic = datetime.datetime.now()
            raise ValueError("You should run ticbegin() first.")
        else:
            self.toc_list.append(self.toc - self.tic)
            self.single_seconds = self.toc - self.tic
            self.wholetime += (self.toc - self.tic).total_seconds()

    def cleartotal(self):
        self.tic = 0
        self.toc = 0
        self.toc_list = []

    def clearhms(self):
        self.hours = 0
        self.minutes = 0
        self.seconds = 0

    def cleartotals(self):
        self.total_seconds = 0

    def caculatetime(self):  # 这是计算列表中的时间差总和的方法
        # 根据开始和结束时间计算时间差，时分秒。
        # 由于 datetime.timedelta() 对象没有提供直接获取时分秒的方法，所以需要自己计算
        # 时分秒的总秒数，然后再转换成时分秒的形式
        for i in self.toc_list:
            self.total_seconds = i.total_seconds()
        self.hours = int(self.total_seconds // 3600)
        self.minutes = int((self.total_seconds - self.hours * 3600) // 60)
        self.seconds = int(self.total_seconds - self.hours * 3600 - self.minutes * 60)
        # self.cleartotal()
        return self.hours, self.minutes, self.seconds

    def caculatewhole(self):
        # 计算总时间
        self.wholehours = int(self.wholetime // 3600)
        self.wholemintues = int((self.wholetime - self.wholehours * 3600) // 60)
        self.wholeseconds = int(self.wholetime - self.wholehours * 3600 - self.wholemintues * 60)

    def printtime(self, content: str, iswhole: bool = False):
        # 打印总时间
        if not iswhole:
            self.caculatetime()
            content = content + "{} hours {} minutes {} seconds."
            print(content.format(self.hours, self.minutes, self.seconds))
        else:
            self.caculatewhole()
            content = content + "{} hours {} minutes {} seconds."
            print(content.format(self.wholehours, self.wholemintues, self.wholeseconds))

    # 计算剩余时间
    def printlefttime(self, epoch: int, total_epoch: int):
        self.caculatetime()
        left_epoch = total_epoch - epoch
        # 应该用left_epoch乘以过去五轮的平均时间；
        # 但是由于这里的时间差是每一轮的时间差，所以直接用这个时间差乘以剩余轮数
        left_seconds = self.total_seconds * left_epoch


        left_hours = int(left_seconds // 3600)
        left_minutes = int((left_seconds - left_hours * 3600) // 60)
        left_seconds = int(left_seconds - left_hours * 3600 - left_minutes * 60)
        content = "The left time is: {} hours {} minutes {} seconds."
        print(content.format(left_hours, left_minutes, left_seconds))



if __name__ == '__main__':
    # 循环测试
    # t = TicToc()
    # for i in range(1):
    #     t.ticbegin()
    #     time.sleep(1)
    #     t.ticend()
    #     t.printtime("The total time is: ")
    #
    # for i in range(2):
    #     t.ticbegin()
    #     time.sleep(1)
    #     t.ticend()
    # t.printtime("The total time is: ")
    # t.cleartotals()
    #
    # for i in range(1):
    #     t.ticbegin()
    #     time.sleep(1)
    #     t.ticend()
    # t.printtime("The total time is: ")

    # 双循环测试
    t1 = TicToc()
    te = TicToc()
    for i in range(10):
        t1.ticbegin()
        te.ticbegin()
        for j in range(1):
            time.sleep(1)
        t1.ticend()

        te.ticend()
        te.printtime("The whole epoch time is: ")

        te.printlefttime(i, 10)

    t1.printtime("The total epoch time is: ", iswhole=True)











