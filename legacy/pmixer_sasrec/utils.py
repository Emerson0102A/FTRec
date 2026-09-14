import sys
import copy
import torch
import random
import numpy as np
from collections import defaultdict
from multiprocessing import Process, Queue

#数据读取
def build_index(dataset_name):

    ui_mat = np.loadtxt('data/%s.txt' % dataset_name, dtype=np.int32)

    n_users = ui_mat[:, 0].max()
    n_items = ui_mat[:, 1].max()

    #res = [[] for _ in range(3 + 1)]
    # 结果：res == [[], [], [], []]  # 4 个独立的空列表


    u2i_index = [[] for _ in range(n_users + 1)] #建立n_users+1个空列表
    i2u_index = [[] for _ in range(n_items + 1)]

    #建立用户-物品和物品-用户的倒排索引
    #例如
    #1: [10,11,12]
    #2: [7]
    for ui_pair in ui_mat:
        u2i_index[ui_pair[0]].append(ui_pair[1])
        i2u_index[ui_pair[1]].append(ui_pair[0])
    
    return u2i_index, i2u_index

#负采样
# sampler for batch generation
def random_neq(l, r, s):
    t = np.random.randint(l, r) #在[l,r)范围内采样
    while t in s: #如果采样到的t在集合s中,则重新采样
        t = np.random.randint(l, r)
    return t

#造一个batch
def sample_function(user_train, usernum, itemnum, batch_size, maxlen, result_queue, SEED):
    def sample(uid):

        # uid = np.random.randint(1, usernum + 1)
        while len(user_train[uid]) <= 1: uid = np.random.randint(1, usernum + 1) #确保用户至少有两个交互记录

        seq = np.zeros([maxlen], dtype=np.int32) #生成长度为maxlen的全0序列
        pos = np.zeros([maxlen], dtype=np.int32)
        neg = np.zeros([maxlen], dtype=np.int32)
        nxt = user_train[uid][-1]   #用户最后一次交互的物品ID
        idx = maxlen - 1          #从序列最后一个位置开始填充

        ts = set(user_train[uid])
        for i in reversed(user_train[uid][:-1]):#从开头到倒数第二个交互记录
            seq[idx] = i
            pos[idx] = nxt #正样本就是下一个物品
            neg[idx] = random_neq(1, itemnum + 1, ts) #在[1,itemnum+1)范围内采样
            nxt = i
            idx -= 1
            if idx == -1: break

        return (uid, seq, pos, neg)

    np.random.seed(SEED) #固定随机种子，保证采样结果可复现
    uids = np.arange(1, usernum+1, dtype=np.int32) #所有用户ID列表
    counter = 0 #采样计数器，用来在uids中循环采样
    while True:
        if counter % usernum == 0: #每个用户的样本都被使用过一次
            np.random.shuffle(uids) #打乱用户顺序
        one_batch = []
        for i in range(batch_size): 
            one_batch.append(sample(uids[counter % usernum])) #循环使用用户
            counter += 1
        result_queue.put(zip(*one_batch)) 

#多进程数据生产者
class WarpSampler(object):
    def __init__(self, User, usernum, itemnum, batch_size=64, maxlen=10, n_workers=1):
        self.result_queue = Queue(maxsize=n_workers * 10)
        self.processors = []
        for i in range(n_workers):
            self.processors.append(
                Process(target=sample_function, args=(User,
                                                      usernum,
                                                      itemnum,
                                                      batch_size,
                                                      maxlen,
                                                      self.result_queue,
                                                      np.random.randint(2e9)
                                                      )))
            self.processors[-1].daemon = True #设为守护进程，子进程随父进程退出而退出
            self.processors[-1].start() #启动子进程

    def next_batch(self):
        return self.result_queue.get()

    def close(self):
        for p in self.processors:
            p.terminate()
            p.join() #等待子进程退出并回收资源，防止僵尸进程

#留一法切分
# train/val/test data generation
def data_partition(fname):
    usernum = 0
    itemnum = 0
    User = defaultdict(list) #默认值是空列表
    
    # from collections import defaultdict

    # User = defaultdict(list)

    # User['alice'].append(1)   # 不会 KeyError，'alice' 自动变成 []
    # User['alice'].append(2)
    # print(User)  # {'alice': [1, 2]}

    user_train = {}
    user_valid = {}
    user_test = {}
    # assume user/item index starting from 1
    f = open('data/%s.txt' % fname, 'r')
    for line in f:
        u, i = line.rstrip().split(' ')
        u = int(u)
        i = int(i)
        usernum = max(u, usernum)
        itemnum = max(i, itemnum)
        User[u].append(i) 

    for user in User:
        nfeedback = len(User[user])
        if nfeedback < 4:                          # To be rigorous, the training set needs at least two data points to learn
            user_train[user] = User[user]
            user_valid[user] = []
            user_test[user] = []
        else:
            user_train[user] = User[user][:-2]
            user_valid[user] = []
            user_valid[user].append(User[user][-2])
            user_test[user] = []
            user_test[user].append(User[user][-1])
    return [user_train, user_valid, user_test, usernum, itemnum]


#排序评测
# TODO: merge evaluate functions for test and val set
# evaluate on test set
def evaluate(model, dataset, args):
    [train, valid, test, usernum, itemnum] = copy.deepcopy(dataset)# 拷贝后原数据不变

    NDCG = 0.0
    HT = 0.0
    valid_user = 0.0

    if usernum>10000:
        users = random.sample(range(1, usernum + 1), 10000)
    else:
        users = range(1, usernum + 1)
    for u in users:

        if len(train[u]) < 1 or len(test[u]) < 1: continue

        seq = np.zeros([args.maxlen], dtype=np.int32)
        idx = args.maxlen - 1
        seq[idx] = valid[u][0] 
        idx -= 1
        for i in reversed(train[u]):
            seq[idx] = i
            idx -= 1
            if idx == -1: break
        rated = set(train[u]) #用户所有交互过的物品集合
        rated.add(0) #添加0,因为0是padding item，不参与评分
        item_idx = [test[u][0]] #测试集的正样本
        for _ in range(100):
            t = np.random.randint(1, itemnum + 1) #在[1,itemnum+1)范围内采样
            while t in rated: t = np.random.randint(1, itemnum + 1) #负采样
            item_idx.append(t) #负样本

        predictions = -model.predict(*[np.array(l) for l in [[u], [seq], item_idx]]) 
        predictions = predictions[0] # - for 1st argsort DESC

        #argsort():返回数组值从小到大的索引值
        #DESC:降序
        
        rank = predictions.argsort().argsort()[0].item()  # 双重argsort得到降序排名,取出正样本的排名,item_idx[rank] == test[u][0]

        valid_user += 1

        if rank < 10:
            NDCG += 1 / np.log2(rank + 2) #np.log2(rank+2)是DCG的折损因子
            HT += 1
        if valid_user % 100 == 0:
            print('.', end="")
            sys.stdout.flush()

    return NDCG / valid_user, HT / valid_user


# evaluate on val set
def evaluate_valid(model, dataset, args):
    [train, valid, test, usernum, itemnum] = copy.deepcopy(dataset)

    NDCG = 0.0
    valid_user = 0.0
    HT = 0.0
    if usernum>10000:
        users = random.sample(range(1, usernum + 1), 10000)
    else:
        users = range(1, usernum + 1)
    for u in users:
        if len(train[u]) < 1 or len(valid[u]) < 1: continue

        seq = np.zeros([args.maxlen], dtype=np.int32)
        idx = args.maxlen - 1
        for i in reversed(train[u]):
            seq[idx] = i
            idx -= 1
            if idx == -1: break

        rated = set(train[u])
        rated.add(0)
        item_idx = [valid[u][0]]
        for _ in range(100):
            t = np.random.randint(1, itemnum + 1)
            while t in rated: t = np.random.randint(1, itemnum + 1)
            item_idx.append(t)

        predictions = -model.predict(*[np.array(l) for l in [[u], [seq], item_idx]])
        predictions = predictions[0]

        rank = predictions.argsort().argsort()[0].item()

        valid_user += 1

        if rank < 10:
            NDCG += 1 / np.log2(rank + 2)
            HT += 1
        if valid_user % 100 == 0:
            print('.', end="")
            sys.stdout.flush()

    return NDCG / valid_user, HT / valid_user
