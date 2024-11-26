import csv
import os
import numpy as np
import torch
from torch.utils.data import Dataset

epsilon = np.finfo(float).eps


#  myDataset继承Dateset
class myDataset(Dataset):
    def __init__(self, mode, data="./", task="SpokenEEG", recon="Y_mel"):
        self.sample_rate = 8000
        self.n_classes = 13
        self.mode = mode
        self.iter = iter
        self.savedata = data
        self.task = task
        self.recon = recon
        self.max_audio = 32768.0
        self.lenth = len(os.listdir(self.savedata + '/train/Y/'))  # 780 # the number data
        self.lenthtest = len(os.listdir(self.savedata + '/test/Y/'))  # 260
        self.lenthval = len(os.listdir(self.savedata + '/val/Y/'))  # 260

    # get length
    def __len__(self):
        if self.mode == 2:
            return self.lenthval
        elif self.mode == 1:
            return self.lenthtest
        else:
            return self.lenth

    # 获取具体样本
    def __getitem__(self, idx):
        '''
        :param idx:
        :return:
        '''

        if self.mode == 2:
            forder_name = self.savedata + '/val/'
        elif self.mode == 1:
            forder_name = self.savedata + '/test/'
        else:
            forder_name = self.savedata + '/train/'

        # tasks
        # print(forder_name)  # 打印当前文件夹路径
        # print(self.task)  # 打印任务类型
        #  从指定目录中获取一个特定文件的路径
        #  os.listdir用于列出forder_name+self.task+/目录中的所有文件和子目录
        allFileList = os.listdir(forder_name + self.task + "/")
        allFileList.sort()
        file_name = forder_name + self.task + '/' + allFileList[idx]

        # if self.task.find('vec') != -1: # embedding vector
        #     input, avg_input, std_input = self.read_vector_data(file_name)

        if self.task.find('mel') != -1: #判断当前的self.task中是否包含‘mel'
            input, avg_input, std_input = self.read_data(file_name)
        elif self.task.find('Voice') != -1:  # voice
            input, avg_input, std_input = self.read_voice_data(file_name)
        else:  # EEG
            input, avg_input, std_input = self.read_data(file_name)

        # recon target
        allFileList = os.listdir(forder_name + self.recon + "/")
        allFileList.sort()
        file_name = forder_name + self.recon + '/' + allFileList[idx]

        # if self.recon.find('vec') != -1: # embedding vector
        #     target, avg_target, std_target = self.read_vector_data(file_name) 
        if self.recon.find('mel') != -1:
            target, avg_target, std_target = self.read_data(file_name)
        elif self.recon.find('Voice') != -1:  # voice
            target, avg_target, std_target = self.read_voice_data(file_name)
        else:  # EEG
            target, avg_target, std_target = self.read_data(file_name)  # std标准差

        # voice
        allFileList = os.listdir(forder_name + "Voice/")
        allFileList.sort()
        file_name = forder_name + "Voice/" + allFileList[idx]
        voice, _, _ = self.read_voice_data(file_name)
        # voice=[]
        # target label
        allFileList = os.listdir(forder_name + "Y/")
        allFileList.sort()
        file_name = forder_name + 'Y/' + allFileList[idx]

        target_cl, _, _ = self.read_raw_data(file_name)
        target_cl = np.squeeze(target_cl)  # 移除target_cl数组中所有维度为1的轴

        # input/target to tensor
        input = torch.tensor(input, dtype=torch.float32)
        target = torch.tensor(target, dtype=torch.float32)

        return input, target, target_cl, voice, (avg_target, std_target, avg_input, std_input)

    def read_vector_data(self, file_name, n_classes):
        with open(file_name, 'r', newline='') as f:  # 打开file_name指定的CSV文件，将每一行数据追加到data列表中
            lines = csv.reader(f)
            data = []
            for line in lines:
                data.append(line)

        data = np.array(data).astype(np.float32)  # 将data列表转化为numpy数组，并将数据类型转为为float32
        (r, c) = data.shape
        data = np.reshape(data, (n_classes, r // n_classes, c))  # 将数据重塑为（n_classes,r//n_classes,c)

        max_ = np.max(data).astype(np.float32)
        min_ = np.min(data).astype(np.float32)
        avg = (max_ + min_) / 2
        std = (max_ - min_) / 2

        data = np.array((data - avg) / std).astype(np.float32)  # 标准化 (data-avg)/std

        return data, avg, std

    def read_voice_data(self, file_name):
        with open(file_name, 'r', newline='') as f:
            lines = csv.reader(f)
            data = []
            for line in lines:
                data.append(line)
        data = np.array(data).astype(np.float32)

        data = np.array(data / self.max_audio).astype(np.float32)  # 归一化处理
        avg = np.array([0]).astype(np.float32)  # 输出一个浮点数【0.0】数组

        return data, avg, self.max_audio

    def read_data(self, file_name):
        with open(file_name, 'r', newline='') as f:
            lines = csv.reader(f)
            data = []
            for line in lines:
                data.append(line)

        data = np.array(data).astype(np.float32)

        max_ = np.max(data).astype(np.float32)
        min_ = np.min(data).astype(np.float32)
        avg = (max_ + min_) / 2
        std = (max_ - min_) / 2

        data = np.array((data - avg) / std).astype(np.float32)

        return data, avg, std

    def read_raw_data(self, file_name):
        with open(file_name, 'r', newline='') as f:
            lines = csv.reader(f)
            data = []
            for line in lines:
                data.append(line)

        data = np.array(data).astype(np.float32)
        avg = np.array([0]).astype(np.float32)
        std = np.array([1]).astype(np.float32)

        return data, avg, std
