# @Time    : 2020/5/22 11:49
# @Author  : iRinYe
# @Email   : YeYilinCN@outlook.com
# @File    : DeepLin
# @Software: PyCharm

"""
    DeepLin
"""
import math
import random
import warnings
from string import ascii_letters, digits
from time import sleep

import numpy as np
import os
import torch
from sklearn.metrics import roc_curve, precision_recall_curve, auc
from torch.optim import Optimizer
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

# 共有超参数
publicHyperParams = {
    'seed': 1,  # 随机种子
    # jobID: 每次运行代码会生成统一的随机jodID，请勿尝试更改，会严重影响模型训练与测试结果
    'jobID': None,
}


class DeepLing:
    def __init__(self, modelClass, lossFunc, optFunc='RAdam',
                 epoch=10, early_stop=0, batch_size=128, lr=0.001,
                 l1=0, l2=0, weight=None,
                 p_Labels=0, a=0, T1=0, T2=0,
                 isCuda=True, isSelf=False, isMask=False, showLoss=True, opt_level=None):
        """
        初始化DeepLing类
        :param modelClass: 模型的定义，传入PyTorch的网络Class即可
        :param lossFunc: 损失函数
            'CEP':CrossEntropy；
            'MSE':MSELoss;
            或是直接送入'torch.nn.modules.loss'下的损失函数
        :param optFunc: 优化函数
            'RAdam': RAdam（默认）;
            'SGD': SGD;
            'Adam': Adam;
            或是直接送入'torch.optim'下的优化函数
        :param epoch: 没什么好说的
        :param early_stop: 早停的等待次数，当其为0时禁用早停
        :param batch_size: 没什么好说的
        :param lr: learning rate，默认0.001
        :param l1: l1范数的权重，默认为0（不启用）
        :param l2: l2范数的权重，默认为0（不启用）
        :param weight: 网络的初始化权重
        :param p_Labels: 是否使用伪标签进行训练
        :param a: 退火算法相关
        :param T1: 退火算法相关
        :param T2: 退火算法相关

        :param isCuda: 是否使用GPU进行训练
        :param isSelf: 是否是无标签学习
        :param isMask: 在训练过程中是否需要使用MASK屏蔽缺失数据
        :param showLoss: 是否显示loss
        """
        if publicHyperParams['jobID'] is None:
            # 如果jobID缺失，就随机产生新的jobID并更新到Params类中
            publicHyperParams['jobID'] = ''.join(random.sample(ascii_letters + digits, 8))
        self.jobID = publicHyperParams['jobID']
        print('jobID is ''{}'''.format(self.jobID))

        if publicHyperParams['seed'] > 0:
            # 如果seed大于0，那么就尽可能的确保模型的可复现性
            self.__recurrentMode(publicHyperParams['seed'])

        self.model = modelClass()

        if lossFunc == "CEP":
            self.loss_func = torch.nn.CrossEntropyLoss()
        elif lossFunc == "MSE":
            self.loss_func = torch.nn.MSELoss()
        else:
            if 'torch.nn.modules.loss' in str(type(lossFunc)):
                self.loss_func = lossFunc
            else:
                raise RuntimeError('请选择适合的loss函数！')

        self.modelName = self.model.modelName + "_" + self.jobID

        self.isCuda = isCuda
        self.EPOCH = epoch
        self.lr = lr
        self.weight_decay = l2
        self.weight = weight
        self.batch_size = batch_size
        self.l1 = l1
        self.EarlyStop = early_stop

        self.p_Labels = p_Labels
        self.a = a

        self.T1 = T1
        self.T2 = T2

        self.isSelf = isSelf
        self.isMask = isMask
        self.showLoss = showLoss
        self.opt_level = opt_level

        if optFunc == 'RAdam':
            self.optimizer = RAdam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        elif optFunc == 'Adam':
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        elif optFunc == 'SGD':
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        else:
            if 'torch.optim' in str(type(optFunc)):
                self.loss_func = lossFunc
            else:
                raise RuntimeError('请选择适合的优化函数！')

        if self.isSelf and self.p_Labels:
            raise RuntimeError('伪标签（半监督）训练无法在无监督模式下工作！')

    def __recurrentMode(self, seed=1):
        """
        启动复现模式，调用该方法将尽可能的确保模型的可复现性
        :param seed: 随机种子，默认为1
        :return: 无
        """
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.deterministic = True

    def cudaSTH(self, *STH):
        """
        判定并将STH中的item移入显卡或是CPU中
        :param STH: 若干可放入cuda中的数据
        :return: 返回尝试放入后的数据
        """
        if torch.cuda.is_available() and self.isCuda:
            print("Working on GPU Mode")
            cudaTH = ()
            for item in STH:
                if item is not None:
                    cudaTH += (item.cuda(),)
                else:
                    cudaTH += (item,)

        else:
            cudaTH = ()
            for item in STH:
                if item is not None:
                    if type(item) != list:
                        cudaTH += (item.cpu(),)
                    else:
                        tmp = []
                        for tmp_item in item:
                            tmp.append(tmp_item)
                        cudaTH += (tmp,)
                else:
                    cudaTH += (item,)

        if len(cudaTH) == 1:
            return cudaTH[0]
        else:
            return cudaTH

    def getTrainingDataLoader(self, x, y=None, isShuffle=True, preShuffle=False):
        if y is None:
            warnings.warn("未提供Label数据，已自动切换到无监督学习模式.", DeprecationWarning)
            self.loss_func = torch.nn.MSELoss()
            self.isSelf = True

        if y is not None and self.isSelf:
            warnings.warn("已提供Label数据，但仍选择无监督学习模式，已自动切换到有监督学习模式.", DeprecationWarning)
            self.isSelf = False

        self.__getDataLoader(x, y, isShuffle, preShuffle, isTraining=True)

    def getTestDataLoader(self, x, y=None, isShuffle=False, preShuffle=False):
        self.__getDataLoader(x, y, isShuffle, preShuffle, isTraining=False)

    def __getDataLoader(self, x, y=None, isShuffle=True, preShuffle=False, isTraining=True):
        """
        获取打包好的DataLoader
        :type x: array 数据的特征部分
        :type y: array 数据的label部分
        :type batch_size: int batch size
        :type isShuffle: bool 是否需要打乱
        :param preShuffle: 是否需要预先打乱（自动关闭isShuffle）
        :param isTraining: 是否是训练数据
        """

        x = np.array(x, dtype='float')

        if isTraining is False and isShuffle:
            warnings.warn("测试数据被打乱，请确认是否是你需要的，Shuffle已禁用", DeprecationWarning)
            isShuffle = False
            preShuffle = False

        if preShuffle:
            # 预先打乱的话，就不用pytorch的打乱了
            shuffle_ix = np.random.permutation(np.arange(len(x)))
            x = x[shuffle_ix]
            if y is not None:
                y = np.array(y, dtype='float')
                y = y[shuffle_ix]
            isShuffle = False

        x_tenser = torch.Tensor.float(torch.from_numpy(x))

        if y is not None:
            y = np.array(y, dtype='float')
            y_tenser = torch.Tensor.long(torch.from_numpy(y))
            dataset = TensorDataset(x_tenser, y_tenser)
        else:
            dataset = TensorDataset(x_tenser)
        dataloader = DataLoader(dataset=dataset, batch_size=self.batch_size, shuffle=isShuffle, drop_last=False)

        if isTraining:
            self.dl_training = dataloader
        else:
            self.dl_test = dataloader

    def __trainProcess(self, batch_data, t=None):
        model = self.model
        model.train()
        b_x, b_y = self.cudaSTH(batch_data)

        if self.isMask:
            mask = b_x[:, int(b_x.size()[1] / 2):]
            b_x = b_x[:, :int(b_x.size()[1] / 2)]

        pred = model(b_x)

        if b_y is None:
            if self.isMask:
                loss = self.loss_func(pred[1] * mask, b_x * mask)  # 自动编码器(带MASK)用
            else:
                loss = self.loss_func(pred[1], b_x)  # 自动编码器（无MASK）用
        else:
            if self.p_Labels == 0:
                # 不使用伪标签进行训练
                try:
                    loss = self.loss_func(pred, b_y.float().view(-1, 1))
                except:
                    loss = self.loss_func(pred, b_y.long().view(-1))
            else:
                # 使用伪标签进行训练
                p_Labels_index = b_y[:, 1] == -1  # 伪标签样本索引
                t_Labels_index = b_y[:, 1] != -1  # 真实标签样本索引

                if sum(t_Labels_index) != 0:
                    # 如果存在真实标签样本
                    p_loss = self.loss_func(pred[p_Labels_index],
                                            b_y[p_Labels_index][:, 0].float().view(-1, 1))  # 伪标签损失
                    t_loss = self.loss_func(pred[t_Labels_index],
                                            b_y[t_Labels_index][:, 0].float().view(-1, 1))  # 真实标签损失
                    a_pie = self.SAA(t)
                    loss = t_loss + a_pie * p_loss
                else:
                    p_loss = self.loss_func(pred[p_Labels_index],
                                            b_y[p_Labels_index][:, 0].float().view(-1, 1))  # 伪标签损失
                    a_pie = self.SAA(t)
                    loss = a_pie * p_loss

                b_y[p_Labels_index][:, 0] = pred[p_Labels_index].argmax(dim=1)

        if self.l1 != 0:
            # lamda不为0时计算l1范式
            regularization_loss = 0
            for param in model.parameters():
                regularization_loss += torch.sum(abs(param))

            loss = loss + self.l1 * regularization_loss

        self.optimizer.zero_grad()

        # if self.opt_level is not None:
        #     # todo 完成混合精度训练代码
        #     with amp.scale_loss(loss, optimizer) as scaled_loss:
        #         scaled_loss.backward()

        loss.backward()
        self.optimizer.step()

        return loss.item(), b_y

    def train(self):
        """
        训练模型
        """

        if self.weight is not None:
            self.weight = torch.from_numpy(np.array(self.weight)).float()

        if self.EarlyStop != 0:
            ES = EarlyStopping(patience=self.EarlyStop, path='{}.pt'.format(self.modelName), verbose=self.showLoss)
        else:
            ES = None

        self.model, self.weight = self.cudaSTH(self.model, self.weight)

        # if self.opt_level is not None:
        #     # todo 完成混合精度训练代码
        #     model, optimizer = amp.initialize(model, optimizer, opt_level="O0")

        newDataset = None
        for epoch in range(self.EPOCH):
            new_epoch_loss = 0
            step = 0

            for batch_data in self.dl_training if self.showLoss else tqdm(self.dl_training):
                loss, b_y = self.__trainProcess(batch_data)
                new_epoch_loss += loss

                if self.p_Labels == 1:
                    if newDataset is None:
                        newDataset = b_y
                    else:
                        newDataset = torch.cat((newDataset, b_y))
            step += 1

            if self.showLoss is True:
                tqdm.write("epoch:{}/{} loss: {}".format(epoch + 1, self.EPOCH, round(new_epoch_loss / step, 4)))

            if ES is not None:
                ES(round(new_epoch_loss, 3), self.model)
                if ES.early_stop:
                    print("Early Stop in {}!!!".format(epoch))
                    break

            if self.p_Labels == 1:
                self.dl_training.dataset.tensors[1].data = newDataset

    def test(self):
        """
        训练模型
        :param model: PyTorch模型
        :param dataloader: dataloader
        :return: 预测结果矩阵(array)
        """
        if self.EarlyStop > 0:
            try:
                self.model.load_state_dict(torch.load('tmp/{}.pt'.format(self.modelName)))
                # os.remove('tmp/{}.pt'.format(model.modelName))
                print("已读取{}_Early_Stop模型".format(self.modelName))
            except:
                print("未读取{}_Early_Stop模型，或直接使用Cache模型".format(self.modelName))

        model = self.cudaSTH(self.model)

        pred = None

        with torch.no_grad():
            if self.isSelf is False:
                for b_x, _ in tqdm(self.dl_test):
                    model.eval()

                    b_x = self.cudaSTH(b_x)

                    temp = torch.Tensor.softmax(model(b_x), dim=1).cpu().numpy()
                    if pred is None:
                        pred = temp
                    else:
                        pred = np.vstack((pred, temp))
            else:
                for b_x in tqdm(self.dl_test):
                    model.eval()
                    b_x = b_x[0]
                    if self.isMask:
                        b_x = b_x[:, :int(b_x.size()[1] / 2)]

                    b_x = self.cudaSTH(b_x)

                    if self.isSelf:
                        temp = model(b_x)[0].cpu().numpy()
                    else:
                        temp = model(b_x).cpu().numpy()

                    temp = temp.reshape(temp.shape[0], temp.shape[1])
                    if pred is None:
                        pred = temp
                    else:
                        pred = np.vstack((pred, temp))

        return pred[:, 1]

    def __SAA(self, t):
        """
        模拟退火算法
        :param t: 当前的时间
        :return: 模拟退火权重
        """
        T1 = self.T1  # 第一阶段的时间
        T2 = self.T2  # 第二阶段的时间
        a = self.a  # 模拟退火权重

        if t < T1:
            return 0
        elif T1 <= t <= T2:
            return (t - T1) / (T2 - T1) * a
        else:
            return a

    def score(self, true, pred):
        """
        计算auc与aupr值并返回
        :param true: 二元标签向量, 1为真0为假,
        :param pred: 预测结果, 形状与true相同
        :return: auc_value与aupr_value
        """
        sleep(1)
        true, pred = np.array(true), np.array(pred)
        fpr, tpr, _ = roc_curve(true == 1, pred)
        p, r, _ = precision_recall_curve(true == 1, pred)
        auc_value = auc(fpr, tpr)
        aupr_value = auc(r, p)

        print("AUC: {}, AUPR: {}".format(round(auc_value, 8), round(aupr_value, 8)))


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, patience=7, verbose=True, delta=0, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = "tmp/" + path
        self.trace_func = trace_func

        try:
            os.makedirs("tmp")
        except:
            pass

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score <= self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                self.trace_func(f'      EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        """
        Saves model when validation loss decrease.
        """
        if self.verbose:
            self.trace_func(
                f'      Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


class RAdam(Optimizer):

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, degenerated_to_sgd=True):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))

        self.degenerated_to_sgd = degenerated_to_sgd
        if isinstance(params, (list, tuple)) and len(params) > 0 and isinstance(params[0], dict):
            for param in params:
                if 'betas' in param and (param['betas'][0] != betas[0] or param['betas'][1] != betas[1]):
                    param['buffer'] = [[None, None, None] for _ in range(10)]
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
                        buffer=[[None, None, None] for _ in range(10)])
        super(RAdam, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(RAdam, self).__setstate__(state)

    def step(self, closure=None):

        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data.float()
                if grad.is_sparse:
                    raise RuntimeError('RAdam does not support sparse gradients')

                p_data_fp32 = p.data.float()

                state = self.state[p]

                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p_data_fp32)
                    state['exp_avg_sq'] = torch.zeros_like(p_data_fp32)
                else:
                    state['exp_avg'] = state['exp_avg'].type_as(p_data_fp32)
                    state['exp_avg_sq'] = state['exp_avg_sq'].type_as(p_data_fp32)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                exp_avg.mul_(beta1).add_(1 - beta1, grad)

                state['step'] += 1
                buffered = group['buffer'][int(state['step'] % 10)]
                if state['step'] == buffered[0]:
                    N_sma, step_size = buffered[1], buffered[2]
                else:
                    buffered[0] = state['step']
                    beta2_t = beta2 ** state['step']
                    N_sma_max = 2 / (1 - beta2) - 1
                    N_sma = N_sma_max - 2 * state['step'] * beta2_t / (1 - beta2_t)
                    buffered[1] = N_sma

                    # more conservative since it's an approximated value
                    if N_sma >= 5:
                        step_size = math.sqrt(
                            (1 - beta2_t) * (N_sma - 4) / (N_sma_max - 4) * (N_sma - 2) / N_sma * N_sma_max / (
                                    N_sma_max - 2)) / (1 - beta1 ** state['step'])
                    elif self.degenerated_to_sgd:
                        step_size = 1.0 / (1 - beta1 ** state['step'])
                    else:
                        step_size = -1
                    buffered[2] = step_size

                # more conservative since it's an approximated value
                if N_sma >= 5:
                    if group['weight_decay'] != 0:
                        p_data_fp32.add_(-group['weight_decay'] * group['lr'], p_data_fp32)
                    denom = exp_avg_sq.sqrt().add_(group['eps'])
                    p_data_fp32.addcdiv_(-step_size * group['lr'], exp_avg, denom)
                    p.data.copy_(p_data_fp32)
                elif step_size > 0:
                    if group['weight_decay'] != 0:
                        p_data_fp32.add_(-group['weight_decay'] * group['lr'], p_data_fp32)
                    p_data_fp32.add_(-step_size * group['lr'], exp_avg)
                    p.data.copy_(p_data_fp32)

        return loss


class PlainRAdam(Optimizer):

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, degenerated_to_sgd=True):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))

        self.degenerated_to_sgd = degenerated_to_sgd
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)

        super(PlainRAdam, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(PlainRAdam, self).__setstate__(state)

    def step(self, closure=None):

        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data.float()
                if grad.is_sparse:
                    raise RuntimeError('RAdam does not support sparse gradients')

                p_data_fp32 = p.data.float()

                state = self.state[p]

                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p_data_fp32)
                    state['exp_avg_sq'] = torch.zeros_like(p_data_fp32)
                else:
                    state['exp_avg'] = state['exp_avg'].type_as(p_data_fp32)
                    state['exp_avg_sq'] = state['exp_avg_sq'].type_as(p_data_fp32)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                exp_avg.mul_(beta1).add_(1 - beta1, grad)

                state['step'] += 1
                beta2_t = beta2 ** state['step']
                N_sma_max = 2 / (1 - beta2) - 1
                N_sma = N_sma_max - 2 * state['step'] * beta2_t / (1 - beta2_t)

                # more conservative since it's an approximated value
                if N_sma >= 5:
                    if group['weight_decay'] != 0:
                        p_data_fp32.add_(-group['weight_decay'] * group['lr'], p_data_fp32)
                    step_size = group['lr'] * math.sqrt(
                        (1 - beta2_t) * (N_sma - 4) / (N_sma_max - 4) * (N_sma - 2) / N_sma * N_sma_max / (
                                N_sma_max - 2)) / (1 - beta1 ** state['step'])
                    denom = exp_avg_sq.sqrt().add_(group['eps'])
                    p_data_fp32.addcdiv_(-step_size, exp_avg, denom)
                    p.data.copy_(p_data_fp32)
                elif self.degenerated_to_sgd:
                    if group['weight_decay'] != 0:
                        p_data_fp32.add_(-group['weight_decay'] * group['lr'], p_data_fp32)
                    step_size = group['lr'] / (1 - beta1 ** state['step'])
                    p_data_fp32.add_(-step_size, exp_avg)
                    p.data.copy_(p_data_fp32)

        return loss


class AdamW(Optimizer):

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, warmup=0):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))

        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, warmup=warmup)
        super(AdamW, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(AdamW, self).__setstate__(state)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data.float()
                if grad.is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')

                p_data_fp32 = p.data.float()

                state = self.state[p]

                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p_data_fp32)
                    state['exp_avg_sq'] = torch.zeros_like(p_data_fp32)
                else:
                    state['exp_avg'] = state['exp_avg'].type_as(p_data_fp32)
                    state['exp_avg_sq'] = state['exp_avg_sq'].type_as(p_data_fp32)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                exp_avg.mul_(beta1).add_(1 - beta1, grad)

                denom = exp_avg_sq.sqrt().add_(group['eps'])
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']

                if group['warmup'] > state['step']:
                    scheduled_lr = 1e-8 + state['step'] * group['lr'] / group['warmup']
                else:
                    scheduled_lr = group['lr']
                step_size = scheduled_lr * math.sqrt(bias_correction2) / bias_correction1
                if group['weight_decay'] != 0:
                    p_data_fp32.add_(-group['weight_decay'] * scheduled_lr, p_data_fp32)
                p_data_fp32.addcdiv_(-step_size, exp_avg, denom)
                p.data.copy_(p_data_fp32)
        return loss
