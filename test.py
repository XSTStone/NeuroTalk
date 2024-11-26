import torch
import numpy as np
from glob import glob

device = torch.device('cuda')  # 可以改为GPU

model, decoder, utils = torch.hub.load(repo_or_dir='snakers4/silero-models',
                                       model='silero_stt',
                                       language='es',  # 也可以选择 'de', 'es'
                                       device=device)

(read_batch, split_into_batches,
 read_audio, prepare_model_input) = utils  # 获取必要的函数

audio_folder = 'dataset\\sub1\\val\\Voice\\'  # 替换为你的本地文件夹路径

# 使用glob获取所有CSV文件的路径
csv_files = glob(f'{audio_folder}/*.csv')


# 读取CSV文件并将其转换为音频信号
def load_audio_from_csv(csv_file):
    # 使用numpy加载CSV文件，假设数据没有列标题
    data = np.loadtxt(csv_file, delimiter=',')

    # 将其转换为PyTorch张量并调整维度
    audio_tensor = torch.tensor(data, dtype=torch.float32).unsqueeze(0)  # 添加批次维度
    return audio_tensor


# 加载一个批次的CSV文件并处理
batches = split_into_batches(csv_files, batch_size=10)

# 选择第一个批次中的文件
csv_file = batches[0][0]

# 使用自定义的加载函数加载CSV文件并将其转换为音频信号
audio_tensor = load_audio_from_csv(csv_file)

# 准备输入模型
input = prepare_model_input(audio_tensor, device=device)

# 将输入传递给模型并获取输出
output1 = model(input)
output2 = model(input)

print('hello')

# 打印每个输出的解码结果
# for example in output1:
#     print(decoder(example.cpu()))
