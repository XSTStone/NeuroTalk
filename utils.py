
import numpy as np
import matplotlib.pyplot as plt
import os
import torch
import glob
from torch.nn.utils import weight_norm

def audio_denorm(data):
    max_audio = 32768.0
    
    data = np.array(data * max_audio).astype(np.float32)
       
    return data


# def data_denorm(data, avg, std):
#
#     std = std.type(torch.cuda.FloatTensor)
#     avg = avg.type(torch.cuda.FloatTensor) #将std avg转换为浮点张量，以便在GP上运行
#
#     # if std == 0, change to 1.0 for nothing happen
#     std = torch.where(std==torch.tensor(0,dtype=torch.float32).cuda(), torch.tensor(1,dtype=torch.float32).cuda(), std)
#
#     # change the size of std and avg repeat:将std.avg的最后一个维度重复data的第二维和第三维的大小，生成张量（C,N,T），再将最后一个维度移到前面去
#     std = torch.permute(std.repeat(data.shape[1],data.shape[2],1),[2,0,1])
#     avg = torch.permute(avg.repeat(data.shape[1],data.shape[2],1),[2,0,1])
#
#     data = torch.mul(data, std) + avg  # data=data*std+avg反归一化
#
#     return data

# Stone
def data_denorm(data, avg, std):
    # 将std和avg转换为float32类型张量，确保它们是在CPU上
    std = std.type(torch.float32)  # 修改为float32类型，而不是cuda.FloatTensor
    avg = avg.type(torch.float32)  # 同样修改为float32类型

    # 如果 std 为 0，将其修改为 1.0，避免无效的除法
    std = torch.where(std == torch.tensor(0, dtype=torch.float32), torch.tensor(1, dtype=torch.float32), std)

    # 将std和avg的最后一个维度扩展为 data 的形状，并进行维度交换
    std = torch.permute(std.repeat(data.shape[1], data.shape[2], 1), [2, 0, 1])
    avg = torch.permute(avg.repeat(data.shape[1], data.shape[2], 1), [2, 0, 1])

    # 执行反归一化操作
    data = torch.mul(data, std) + avg  # data = data * std + avg

    return data


def plot_spectrogram(spectrogram):
    fig, ax = plt.subplots(figsize=(10, 2))
    im = ax.imshow(spectrogram, aspect="auto", origin="lower",
                   interpolation='none')
    plt.colorbar(im, ax=ax)
    fig.canvas.draw()
    plt.close()

    return fig
    
def imgSave(dir, file_name):
    if not os.path.exists(dir):
        os.mkdir(dir)
    plt.tight_layout()
    plt.savefig(dir + file_name)
    plt.clf()


def word_index(word_label, bundle):
    labels_ = ''.join(list(bundle.get_labels())) #包含了所有可能字符的字符串
    word_indices = np.zeros((len(word_label), 15), dtype=np.int64) #二维
    word_length = np.zeros((len(word_label), ), dtype=np.int64) #一维，存放每个word_label的长度
    for w in range(len(word_label)): #遍历word_label中的每一个单词
        word = word_label[w] #word为当前处理的单词
        label_idx = []
        for ww in range(len(word)):
            label_idx.append(labels_.find(word[ww]))
        word_indices[w,:len(label_idx)] = torch.tensor(label_idx)
        word_length[w] = len(label_idx)
        
    return word_indices, word_length


######################################################################
############                  HiFiGAN                   ##############
######################################################################
def init_weights(m, mean=0.0, std=0.01):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(mean, std)


def apply_weight_norm(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        weight_norm(m)


def get_padding(kernel_size, dilation=1):
    return int((kernel_size*dilation - dilation)/2)



