import copy
import os
import torch
from models import models as networks
from models.models_HiFi import Generator as model_HiFi
from modules import DTW_align, GreedyCTCDecoder, AttrDict, RMSELoss, save_checkpoint
from modules import mel2wav_vocoder, perform_STT
from utils import data_denorm, word_index
import torch.nn as nn
import torch.nn.functional as F
from NeuroTalkDataset import myDataset
import time
import torch.optim.lr_scheduler
import numpy as np
import torchaudio
from torchmetrics import CharErrorRate
import json
import argparse
import wavio
from torch.utils.tensorboard import SummaryWriter
import json
import requests
from collections import defaultdict

from glob import glob

    
def train(args, train_loader, models, criterions, optimizers, epoch, prepare_model_input, generation_counts=None, empty_counts=None,trainValid=True, inference=False):
    '''
    :param args: general arguments
    :param train_loader: loaded for training/validation/test dataset
    :param model: model
    :param criterion: loss function
    :param optimizer: optimization algo, such as ADAM or SGD
    :param epoch: epoch number
    :return: losses
    '''
    (optimizer_g, optimizer_d) = optimizers
    
    # switch to train mode
    assert type(models) == tuple, "More than two models should be inputed (generator and discriminator)"

    epoch_loss_g = []
    epoch_loss_d = []
    
    epoch_acc_g = []
    epoch_acc_d = []
    
    epoch_loss_g_ns = []
    epoch_loss_d_ns = []
    
    epoch_acc_g_ns = []
    epoch_acc_d_ns = []

    total_batches = len(train_loader)
    
    for i, (input, target, target_cl, voice, data_info) in enumerate(train_loader):    

        print("\rBatch [%5d / %5d]"%(i,total_batches), sep=' ', end='', flush=True)
        
        input = input.cuda()
        target = target.cuda()
        padding = (0, 14)  # 在最后一个维度填充 14 个
        target = torch.nn.functional.pad(target, padding)
        target_cl = target_cl.cuda()
        voice = torch.squeeze(voice,dim=-1).cuda()  # 移除最后一个维度
        labels = torch.argmax(target_cl,dim=1)  # 标注这一批次的样本分别是哪几类 （1维）

        # extract unseen
        idx_unseen=[]
        idx_seen=[]
        for j in range(len(labels)):
            if args.classname[labels[j]] == args.unseen:
                idx_unseen.append(j)
            else:
                idx_seen.append(j)
        
        input_ns = input[idx_unseen]
        target_ns = target[idx_unseen]
        target_cl_ns = target_cl[idx_unseen]
        voice_ns = voice[idx_unseen]
        labels_ns = labels[idx_unseen]
        data_info_ns = [data_info[0][idx_unseen],data_info[1][idx_unseen]]
        
        input = input[idx_seen]
        target = target[idx_seen]
        target_cl = target_cl[idx_seen]
        voice = voice[idx_seen]
        labels = labels[idx_seen]
        data_info = [data_info[0][idx_seen],data_info[1][idx_seen]]

        
        # # need to remove
        # models = (model_g, model_d, vocoder, model_STT, decoder_STT)
        # criterions = (criterion_recon, criterion_ctc, criterion_adv, criterion_cl, CER)
        # trainValid = True
        
        # general training         
        if len(input) != 0:
            # train generator
            mel_out, e_loss_g, e_acc_g, generation_counts, empty_counts = train_G(args,
                                                 input, target, voice, labels,  
                                                 models, criterions, optimizer_g,
                                                 data_info, 
                                                 trainValid,
                                                generation_counts, empty_counts, prepare_model_input=prepare_model_input)
            print("==================================")
            print(e_loss_g, e_acc_g)
            epoch_loss_g.append(e_loss_g)
            epoch_acc_g.append(e_acc_g)
        
            # train discriminator
            e_loss_d, e_acc_d = train_D(args,
                                        mel_out, target, target_cl, labels,
                                        models, criterions, optimizer_d, 
                                        trainValid)
            epoch_loss_d.append(e_loss_d)
            epoch_acc_d.append(e_acc_d)
        
        # Unseen words training
        if len(input_ns) != 0 :
            # Unseen train generator
            mel_out_ns, e_loss_g_ns, e_acc_g_ns = train_G(args, 
                                                          input_ns, target_ns, voice_ns, labels_ns, 
                                                          models, criterions, optimizer_g, 
                                                          data_info_ns,
                                                          False)
            epoch_loss_g_ns.append(e_loss_g_ns)
            epoch_acc_g_ns.append(e_acc_g_ns)
            
            # Unseen train discriminator
            e_loss_d_ns, e_acc_d_ns = train_D(args, 
                                              mel_out_ns, target_ns, target_cl_ns, labels_ns, 
                                              models, criterions, optimizer_d, 
                                              False)
            epoch_loss_d_ns.append(e_loss_d_ns)
            epoch_acc_d_ns.append(e_acc_d_ns)

    epoch_loss_g = np.array(epoch_loss_g)
    epoch_acc_g = np.array(epoch_acc_g)
    epoch_loss_d = np.array(epoch_loss_d)
    epoch_acc_d = np.array(epoch_acc_d)
    
    epoch_loss_g_ns = np.array(epoch_loss_g_ns)
    epoch_acc_g_ns = np.array(epoch_acc_g_ns)
    epoch_loss_d_ns = np.array(epoch_loss_d_ns)
    epoch_acc_d_ns = np.array(epoch_acc_d_ns)
    

    print('\n=======================')
    # print('形状')
    # print(epoch_loss_g.shape)
    # print(sum(epoch_loss_g[:,0]))
    # print(len(epoch_loss_g[:,0]))
    args.loss_g = sum(epoch_loss_g[:,0]) / len(epoch_loss_g[:,0])
    args.loss_g_recon = sum(epoch_loss_g[:,1]) / len(epoch_loss_g[:,1])
    args.loss_g_valid = sum(epoch_loss_g[:,2]) / len(epoch_loss_g[:,2])
    args.loss_g_ctc = sum(epoch_loss_g[:,3]) / len(epoch_loss_g[:,3])
    args.acc_g_valid = sum(epoch_acc_g[:,0]) / len(epoch_acc_g[:,0])
    args.cer_gt = sum(epoch_acc_g[:,1]) / len(epoch_acc_g[:,1])
    print("sum = ", sum(epoch_acc_g[:,1]))
    print("len = ", len(epoch_acc_g[:,1]))
    args.cer_recon = sum(epoch_acc_g[:,2]) / len(epoch_acc_g[:,2])
    
    args.loss_d = sum(epoch_loss_d[:,0]) / len(epoch_loss_d[:,0])
    args.loss_d_valid = sum(epoch_loss_d[:,1]) / len(epoch_loss_d[:,1])
    args.loss_d_cl = sum(epoch_loss_d[:,2]) / len(epoch_loss_d[:,2])
    args.acc_d_real = sum(epoch_acc_d[:,0]) / len(epoch_acc_d[:,0])
    args.acc_d_fake = sum(epoch_acc_d[:,1]) / len(epoch_acc_d[:,1])
    args.acc_cl_real = sum(epoch_acc_d[:,2]) / len(epoch_acc_d[:,2])
    args.acc_cl_fake = sum(epoch_acc_d[:,3]) / len(epoch_acc_d[:,3])
    
    # Unseen
    # args.loss_g_ns = sum(epoch_loss_g_ns[:,0]) / len(epoch_loss_g_ns[:,0])
    # args.loss_g_recon_ns = sum(epoch_loss_g_ns[:,1]) / len(epoch_loss_g_ns[:,1])
    # args.loss_g_valid_ns = sum(epoch_loss_g_ns[:,2]) / len(epoch_loss_g_ns[:,2])
    # args.loss_g_ctc_ns = sum(epoch_loss_g_ns[:,3]) / len(epoch_loss_g_ns[:,3])
    # args.acc_g_valid_ns = sum(epoch_acc_g_ns[:,0]) / len(epoch_acc_g_ns[:,0])
    # args.cer_gt_ns = sum(epoch_acc_g_ns[:,1]) / len(epoch_acc_g_ns[:,1])
    # args.cer_recon_ns = sum(epoch_acc_g_ns[:,2]) / len(epoch_acc_g_ns[:,2])
    #
    # args.loss_d_ns = sum(epoch_loss_d_ns[:,0]) / len(epoch_loss_d_ns[:,0])
    # args.loss_d_valid_ns = sum(epoch_loss_d_ns[:,1]) / len(epoch_loss_d_ns[:,1])
    # args.loss_d_cl_ns = sum(epoch_loss_d_ns[:,2]) / len(epoch_loss_d_ns[:,2])
    # args.acc_d_real_ns = sum(epoch_acc_d_ns[:,0]) / len(epoch_acc_d_ns[:,0])
    # args.acc_d_fake_ns = sum(epoch_acc_d_ns[:,1]) / len(epoch_acc_d_ns[:,1])
    # args.acc_cl_real_ns = sum(epoch_acc_d_ns[:,2]) / len(epoch_acc_d_ns[:,2])
    # args.acc_cl_fake_ns = sum(epoch_acc_d_ns[:,3]) / len(epoch_acc_d_ns[:,3])
    
    # tensorboard
    if trainValid:
        tag = 'train'
    else:
        tag = 'valid'

     # 下面这段代码只会在训练过程中执行
    if not inference:
        args.writer.add_scalar("Loss_G/{}".format(tag), args.loss_g, epoch)
        args.writer.add_scalar("CER/{}".format(tag), args.cer_recon, epoch)
        
        args.writer.add_scalar("Loss_G_recon/{}".format(tag), args.loss_g_recon, epoch)
        args.writer.add_scalar("Loss_G_valid/{}".format(tag), args.loss_g_valid, epoch)
        args.writer.add_scalar("Loss_G_ctc/{}".format(tag), args.loss_g_ctc, epoch)
        
        args.writer.add_scalar("ACC_D_real/{}".format(tag), args.acc_d_real, epoch)
        args.writer.add_scalar("ACC_D_fake/{}".format(tag), args.acc_d_fake, epoch)

        # args.writer.add_scalar("Loss_G_unseen/{}".format(tag), args.loss_g_ns, epoch)
        # args.writer.add_scalar("CER_unseen/{}".format(tag), args.cer_recon_ns, epoch)

    print('\n[%3d/%3d] CER-gt: %.4f CER-recon: %.4f / ACC_R: %.4f ACC_F: %.4f / g-RMSE: %.4f g-lossValid: %.4f g-lossCTC: %.4f' 
          % (i, total_batches, 
             args.cer_gt, args.cer_recon, 
             args.acc_d_real, args.acc_d_fake, 
             args.loss_g_recon, args.loss_g_valid, args.loss_g_ctc))
        
        
    return (args.loss_g, args.loss_g_recon, args.loss_g_valid, args.loss_g_ctc, args.acc_g_valid, args.cer_gt, args.cer_recon, 
            args.loss_d, args.acc_d_real, args.acc_d_fake,generation_counts, empty_counts)


def train_G(args, input, target, voice, labels, models, criterions, optimizer_g, data_info, trainValid,generation_counts, empty_counts):

    (model_g, model_d, vocoder, model_STT, decoder_STT) = models
    (criterion_recon, criterion_ctc, criterion_adv, _, CER) =  criterions  # 重构损失，CTC损失，对抗损失，交叉熵，字符错误率
    
    if trainValid:
        # print('================Training Started================')
        model_g.train()
        model_d.train()
        vocoder.train()
        model_STT.train()
        # print('================Training Ended================')
    else:
        model_g.eval()
        model_d.eval()
        vocoder.eval()
        model_STT.eval()

    # Adversarial ground truths 1:real, 0: fake
    valid = torch.ones((len(input), 1), dtype=torch.float32).cuda()  # torch.ones()创建一个张量，形状为（batch_size,1)
    
    ###############################
    # Train Generator
    ###############################
    
    if trainValid:  # 布尔变量，表示为训练还是验证
        for p in model_g.parameters():
            p.requires_grad_(True)   # 可训练 G
        for p in model_d.parameters():
            p.requires_grad_(False)  # 不可训练 D
        for p in vocoder.parameters():
            p.requires_grad_(False)  # freeze vocoder
        for p in model_STT.parameters():
            p.requires_grad_(False)  # freeze model_STT
            
        # set zero grad    
        optimizer_g.zero_grad()  # 将每次训练前的梯度设置为0
        
        # Run Generator
        # print("Hello Stone")
        # print('input size: ', input.shape)
        output = model_g(input)
        # print("Bye Stone")
    else:
        with torch.no_grad():
            # run generator
            output = model_g(input)
    
    # DTW
    # print('output shape: ', output.shape)
    mel_out = output.clone()   # 生成的梅尔频谱图,形状为（3，80，128）
    # print('mel_out size = ', mel_out.shape)
    # print('target size =', target.shape)

    mel_out = DTW_align(mel_out, target)  # target是输入声音经过处理得到的梅尔频谱图
    
    # Run Discriminator
    # print('261, mel_out size = ', mel_out.shape)
    g_valid, _ = model_d(mel_out)  # _表示属于哪个label的概率，g_valid表示真假
    
    # generator loss
    loss_recon = criterion_recon(mel_out, target)  # target为输入声音的梅尔频谱图
    
    # GAN loss
    loss_valid = criterion_adv(g_valid, valid)
    
    # accuracy    args.l_g = h_g.l_g
    acc_g_valid = (g_valid.round() == valid).float().mean()
    
    ###############################
    # Loss from Vocoder - STT
    ###############################
    # out_DTW
    target_denorm = data_denorm(target, data_info[0], data_info[1])
    output_denorm = data_denorm(mel_out, data_info[0], data_info[1])
    # 把模型的输出和标签转换回原始范围

     #标签处理
    gt_label=[] #标签名
    gt_label_idx=[] #索引
    gt_length=[] #长度
    for j in range(len(target)):
        gt_label.append(args.word_label[labels[j].item()])
        gt_label_idx.append(args.word_index[labels[j].item()])
        gt_length.append(args.word_length[labels[j].item()])
    gt_label_idx = torch.tensor(np.array(gt_label_idx),dtype=torch.int64)
    gt_length = torch.tensor(gt_length,dtype=torch.int64)
    
    # target
    ##### HiFi-GAN
    # print('==================Hello ljm==================')
    #  此处是用来将[19, 59, 128]扩充为[19, 80, 128]，扩充方式为60~80全部填充0
    # target_denorm_expanded = torch.zeros(target_denorm.shape[0], 80, target_denorm.shape[2])
    # target_denorm_expanded[:, :60, :] = target_denorm #用target_denorm填充target_denorm_expanded的前59个
    wav_target = vocoder(target_denorm) #用输入的梅尔频谱图重新生成声音
    wav_target = torch.reshape(wav_target, (len(wav_target),wav_target.shape[-1])) #调整形状,降成二维的（batch_size,num_samples)
    
    #### resampling
    wav_target = torchaudio.functional.resample(wav_target, args.sample_rate_mel, args.sample_rate_STT)
    if wav_target.shape[1] !=  voice.shape[1]:
        p = voice.shape[1] - wav_target.shape[1]
        p_s = p//2
        p_e = p-p_s
        wav_target = F.pad(wav_target, (p_s,p_e))
    
    # recon
    ##### HiFi-GAN
    #  此处是用来将[19, 59, 128]扩充为[19, 80, 128]，扩充方式为60~80全部填充0
    # output_denorm_expanded = torch.zeros(output_denorm.shape[0], 80, output_denorm.shape[2])
    # output_denorm_expanded[:, :60, :] = output_denorm
    wav_recon = vocoder(output_denorm)
    wav_recon = torch.reshape(wav_recon, (len(wav_recon),wav_recon.shape[-1]))
    
    #### resampling
    wav_recon = torchaudio.functional.resample(wav_recon, args.sample_rate_mel, args.sample_rate_STT)   
    if wav_recon.shape[1] !=  voice.shape[1]:
        p = voice.shape[1] - wav_recon.shape[1]
        p_s = p//2
        p_e = p-p_s
        wav_recon = F.pad(wav_recon, (p_s,p_e))

    ##### STT Wav2Vec 2.0
    emission_gt, _ = model_STT(voice)
    emission_recon, _ = model_STT(wav_recon)

    #修改
    text_gt = model_STT(wav_target)

    # CTC loss
    input_lengths = torch.full(size=(emission_gt.size(dim=0),), fill_value=emission_gt.size(dim=1), dtype=torch.long)
    emission_recon_ = emission_recon.log_softmax(2)
    loss_ctc = criterion_ctc(emission_recon_.transpose(0, 1), gt_label_idx, input_lengths, gt_length)

    # gt_label_idx表示目标标签，input_lengths表示输入序列的实际长度，gt_length表示实际长度
    
    # total generator loss
    loss_g = args.l_g[0] * loss_recon + args.l_g[1] * loss_valid + args.l_g[2] * loss_ctc

    # Stone
    loss_g = loss_g.requires_grad_()

    # decoder STT
    transcript_gt = []
    transcript_recon = []

    for j in range(len(voice)):
        transcript = decoder_STT(emission_gt[j])
        print(f'transcript: {transcript}')
        transcript_gt.append(transcript)
            
        transcript = decoder_STT(emission_recon[j])
        transcript_recon.append(transcript)

    cer_gt = CER(transcript_gt, gt_label)  # 声音样本经过STT解码的文本与label之间的差异
    cer_recon = CER(transcript_recon, gt_label)  # 重构声音经过解码的文本与label之间的差异
    # print("label:")
    # print(gt_label)
    # print("声音样本经过STT解码之后的文本:")
    # print(transcript_gt)
    # print("重构的文本：")
    # print(transcript_recon)
    temp = {}

    if not trainValid:
        for i in range(len(labels)):
            label = args.word_label[labels[i]]
            generated_text = transcript_gt[i]

            if generated_text == "":
                empty_counts[label] += 1
            else:
                if generated_text in generation_counts[label]:
                    generation_counts[label][generated_text] += 1
                else:
                    generation_counts[label][generated_text] = 1

    if trainValid:
        loss_g.backward() 
        optimizer_g.step()
    
    e_loss_g = (loss_g.item(), loss_recon.item(), loss_valid.item(), loss_ctc.item())
    e_acc_g = (acc_g_valid.item(), cer_gt.item(), cer_recon.item())
    
    return mel_out, e_loss_g, e_acc_g,generation_counts, empty_counts
      
    
def train_D(args, mel_out, target, target_cl, labels, models, criterions, optimizer_d, trainValid):
    
    (_, model_d, _, _, _) = models
    (_, _, criterion_adv, criterion_cl, _) =  criterions

    if trainValid:
        model_d.train()
    else:
        model_d.eval()
    
    # Adversarial ground truths 1:real, 0: fake
    valid = torch.ones((len(mel_out), 1), dtype=torch.float32).cuda()
    fake = torch.zeros((len(mel_out), 1), dtype=torch.float32).cuda()
    
    ###############################
    # Train Discriminator
    ###############################
    
    if trainValid:
        if args.pretrain and args.prefreeze:
            for total_ct, _ in enumerate(model_d.children()):
                ct=0
            for ct, child in enumerate(model_d.children()):
                if ct > total_ct-1: # unfreeze classifier 
                    for param in child.parameters():
                        param.requires_grad = True  # unfreeze D    
        else:
            for p in model_d.parameters():
                p.requires_grad_(True)  # unfreeze D   
                
        # set zero grad
        optimizer_d.zero_grad()

    # run model cl
    real_valid, real_cl = model_d(target)
    fake_valid, fake_cl = model_d(mel_out.detach())

    loss_d_real_valid = criterion_adv(real_valid, valid)
    loss_d_fake_valid = criterion_adv(fake_valid, fake)
    loss_d_real_cl = criterion_cl(real_cl, target_cl)
    
    loss_d_valid = 0.5 * (loss_d_real_valid + loss_d_fake_valid)
    loss_d_cl = loss_d_real_cl
    
    loss_d = args.l_d[0] * loss_d_cl + args.l_d[1] * loss_d_valid
    loss_d = loss_d.requires_grad_()
    
    # accuracy
    acc_d_real = (real_valid.round() == valid).float().mean()
    acc_d_fake = (fake_valid.round() == fake).float().mean()
    preds_real = torch.argmax(real_cl,dim=1)
    acc_cl_real = (preds_real == labels).float().mean()
    preds_fake = torch.argmax(fake_cl,dim=1)
    acc_cl_fake = (preds_fake == labels).float().mean()
    
    if trainValid:
        loss_d.backward()
        optimizer_d.step()

    e_loss_d = (loss_d.item(), loss_d_valid.item(), loss_d_cl.item())
    e_acc_d = (acc_d_real.item(), acc_d_fake.item(), acc_cl_real.item(), acc_cl_fake.item())
    
    return e_loss_d, e_acc_d


def saveData(args, test_loader, models, epoch, losses):
    
    model_g = models[0].eval()
    # model_d = models[1].eval()
    vocoder = models[2].eval()
    model_STT = models[3].eval()
    decoder_STT = models[4]

    input, target, target_cl, voice, data_info = next(iter(test_loader))   
    
    input = input.cuda()
    target = target.cuda()
    padding = (0, 14)  # 在最后一个维度填充 14 个
    target = torch.nn.functional.pad(target, padding)
    voice = torch.squeeze(voice,dim=-1).cuda()
    labels = torch.argmax(target_cl,dim=1)    
    
    with torch.no_grad():
        # run the mdoel
        output = model_g(input)
    
    mel_out = DTW_align(output, target)
    output_denorm = data_denorm(mel_out, data_info[0], data_info[1])

    # 扩充
    # output_denorm_expanded = torch.zeros(output_denorm.shape[0], 80, output_denorm.shape[2])
    # output_denorm_expanded[:, :60, :] = output_denorm
    # print('output denorm expanded shape: ', output_denorm_expanded.shape)
    
    wav_recon = mel2wav_vocoder(torch.unsqueeze(output_denorm[0],dim=0), vocoder, 1)
    wav_recon = torch.reshape(wav_recon, (len(wav_recon),wav_recon.shape[-1]))
    
    wav_recon = torchaudio.functional.resample(wav_recon, args.sample_rate_mel, args.sample_rate_STT)  
    if wav_recon.shape[1] !=  voice.shape[1]:
        p = voice.shape[1] - wav_recon.shape[1]
        p_s = p//2
        p_e = p-p_s
        wav_recon = F.pad(wav_recon, (p_s,p_e))
        
    ##### STT Wav2Vec 2.0
    gt_label = args.word_label[labels[0].item()]
    
    transcript_recon = perform_STT(wav_recon, model_STT, decoder_STT, gt_label, 1)
    
    # save
    wav_recon = np.squeeze(wav_recon.cpu().detach().numpy())

    str_tar = args.word_label[labels[0].item()].replace("|", ",")
    str_tar = str_tar.replace(" ", ",")
    str_tar = str_tar.replace("/", "")

    str_pred = transcript_recon[0].replace("|", ",")
    str_pred = str_pred.replace(" ", ",")

    title = "Tar_{}-Pred_{}".format(str_tar, str_pred)
    print('title: {}'.format(title))
    wavio.write(args.savevoice + '/e{}_{}.wav'.format(str(str(epoch)), title), wav_recon, args.sample_rate_STT, sampwidth=1)


def main(args):
    
    device = torch.device(f'cuda:{args.gpuNum[0]}' if torch.cuda.is_available() else "cpu")  # 设置计算设备
    torch.cuda.set_device(device) # change allocation of current GPU
    print ('Current cuda device: {} '.format(torch.cuda.current_device())) # check
    print('The number of available GPU:{}'.format(torch.cuda.device_count()))
    
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True

    # define generator
    config_file = os.path.join(args.model_config, 'config_g.json')
    with open(config_file) as f:
        data = f.read()
    json_config = json.loads(data)  # data是一个json的字符串，将其转化为python对象
    h_g = AttrDict(json_config)  # 将D的配置文件转化为AttrDict对象，可以用属性访问的方式获取字典中的值
    model_g = networks.Generator(h_g).cuda()  # 使用 h_g 中的配置参数初始化一个生成器模型，并将该模型移动到 CUDA 兼容的 GPU 上
    
    args.sample_rate_mel = args.sampling_rate
    
    # define discriminator
    config_file = os.path.join(args.model_config, 'config_d.json')
    with open(config_file) as f:
        data = f.read()
    json_config = json.loads(data)
    h_d = AttrDict(json_config)
    model_d = networks.Discriminator(h_d).cuda()
    
    # vocoder HiFiGAN
    # LJ_FT_T2_V3/generator_v3,   
    config_file = os.path.join(os.path.split(args.vocoder_pre)[0], 'config.json')
    # print('===============config_file for vocoder: {}==============='.format(config_file))
    with open(config_file) as f:
        data = f.read()

    json_config = json.loads(data)
    h = AttrDict(json_config)
    
    vocoder = model_HiFi(h).cuda()  # 预定义的HiFi-GAN
    state_dict_g = torch.load(args.vocoder_pre) #, map_location=args.device)
    vocoder.load_state_dict(state_dict_g['generator'])
    
    # STT Wav2Vec
    bundle = torchaudio.pipelines.HUBERT_ASR_LARGE
    model_STT = bundle.get_model().cuda()
    args.sample_rate_STT = bundle.sample_rate
    decoder_STT = GreedyCTCDecoder(labels=bundle.get_labels())
    args.word_index, args.word_length = word_index(args.word_label, bundle)

    # Parallel setting
    model_g = nn.DataParallel(model_g, device_ids=args.gpuNum)
    model_d = nn.DataParallel(model_d, device_ids=args.gpuNum)
    vocoder = nn.DataParallel(vocoder, device_ids=args.gpuNum)
    model_STT = nn.DataParallel(model_STT, device_ids=args.gpuNum)

    # loss function 定义损失函数和评估指标
    criterion_recon = RMSELoss().cuda()  # 定义均方根误差（RMSE）
    criterion_adv = nn.BCELoss().cuda()  # 定义BCELoss 用于二分类任务
    criterion_ctc = nn.CTCLoss().cuda()  # 定义CTC损失，语音转文本
    criterion_cl = nn.CrossEntropyLoss().cuda()  # 定义交叉熵损失函数，用于多分类
    CER = CharErrorRate().cuda() # 定义字符错误率

    # optimizer
    optimizer_g = torch.optim.AdamW(model_g.parameters(), lr=args.lr_g, betas=(0.8, 0.99), weight_decay=0.01)
    optimizer_d = torch.optim.AdamW(model_d.parameters(), lr=args.lr_d, betas=(0.8, 0.99), weight_decay=0.01)
    # 创建了一个AdamW优化器，优化G和D的参数，lr=args.lr_g学习率，betas动量参数，weight_decay权重衰减因子

    scheduler_g = torch.optim.lr_scheduler.ExponentialLR(optimizer_g, gamma=args.lr_g_decay, last_epoch=-1)
    scheduler_d = torch.optim.lr_scheduler.ExponentialLR(optimizer_d, gamma=args.lr_d_decay, last_epoch=-1)
    #创建指数衰减学习率调度器，gamma=args.lr_g_decay设置衰减因子，last_epoch=-1表示调度器从头开始计算

   # create the directory if not exist
    if not os.path.exists(args.logDir):
        os.mkdir(args.logDir)
        
    subDir = os.path.join(args.logDir, args.sub)  # 将两个路径组成一个完整的路径
    if not os.path.exists(subDir):
        os.mkdir(subDir)        
        
    saveDir = os.path.join(args.logDir, args.sub, args.task)  # task是什么？
    if not os.path.exists(saveDir):
        os.mkdir(saveDir)

    args.savevoice = saveDir + '/epovoice'
    if not os.path.exists(args.savevoice):
        os.mkdir(args.savevoice)

    args.savemodel = saveDir + '/savemodel'
    if not os.path.exists(args.savemodel):
        os.mkdir(args.savemodel)
        
    args.logs = saveDir + '/logs'
    if not os.path.exists(args.logs):
        os.mkdir(args.logs)
    # 创建目录，用于保存模型训练结果。
        
    # Load trained model
    start_epoch = 0
    if args.pretrain:
        # print('arg.pretrain = ', args.pretrain)
        loc_g = os.path.join(args.trained_model, args.sub, 'BEST_checkpoint_g.pt')
        loc_d = os.path.join(args.trained_model, args.sub, 'BEST_checkpoint_d.pt')

        if os.path.isfile(loc_g): #  检查文件是否存在
            print("=> loading checkpoint '{}'".format(loc_g))
            checkpoint_g = torch.load(loc_g, map_location='cpu')  # 加载检查点
            model_g.load_state_dict(checkpoint_g['state_dict'])  # 恢复
        else:
            print("=> no checkpoint found at '{}'".format(loc_g))

        if os.path.isfile(loc_d):
            print("=> loading checkpoint '{}'".format(loc_d))
            checkpoint_d = torch.load(loc_d, map_location='cpu')
            model_d.load_state_dict(checkpoint_d['state_dict'])
        else:
            print("=> no checkpoint found at '{}'".format(loc_d))

    if args.resume:
        loc_g = os.path.join(args.savemodel, 'checkpoint_g.pt')
        loc_d = os.path.join(args.savemodel, 'checkpoint_d.pt')

        if os.path.isfile(loc_g):
            print("=> loading checkpoint '{}'".format(loc_g))
            checkpoint_g = torch.load(loc_g, map_location='cpu')
            model_g.load_state_dict(checkpoint_g['state_dict'])
            start_epoch = checkpoint_g['epoch'] + 1  # epoch 训练周期
        else:
            print("=> no checkpoint found at '{}'".format(loc_g))

        if os.path.isfile(loc_d):
            print("=> loading checkpoint '{}'".format(loc_d))
            checkpoint_d = torch.load(loc_d, map_location='cpu')
            model_d.load_state_dict(checkpoint_d['state_dict'])
        else:
            print("=> no checkpoint found at '{}'".format(loc_d))

    # Tensorboard setting 将训练过程中的数据记录到指定的日志目录
    args.writer = SummaryWriter(args.logs)
    
    # Data loader define
    # set 随机数生成器
    generator = torch.Generator().manual_seed(args.seed)

    trainset = myDataset(mode=0, data=args.dataLoc+'/'+args.sub, task=args.task, recon=args.recon)
    # task=spokenEEG,recon=Y_mel,data='dataset+sub1'
    train_loader = torch.utils.data.DataLoader(
        trainset, batch_size=args.batch_size, shuffle=True, generator=generator, num_workers=4*len(args.gpuNum), pin_memory=True)
    #  batch_size 每一批次多少样本  shuffle=True 在每一周期开始前对样本打乱  num_workers加载子进程数量
    valset = myDataset(mode=2, data=args.dataLoc+'/'+args.sub, task=args.task, recon=args.recon)
    val_loader = torch.utils.data.DataLoader(
        valset, batch_size=args.batch_size, shuffle=False, generator=generator, num_workers=4*len(args.gpuNum), pin_memory=True)

    epoch = start_epoch  # start_epoch为0
    lr_g = 0
    lr_d = 0
    best_loss = 1000
    is_best = False
    epochs_since_improvement = 0

    # start
    
    for epoch in range(start_epoch, args.max_epochs):
        
        start_time = time.time()
        # param_groups是优化器中的字典列表，访问优化器中的学习率的语句为：

        generation_counts = {cls: {} for cls in args.word_label}
        empty_counts = {cls: 0 for cls in args.word_label}

        for param_group in optimizer_g.param_groups:
            lr_g = param_group['lr']
        for param_group in optimizer_d.param_groups:
            lr_d = param_group['lr']

        # 更新scheduler调整lr
        scheduler_g.step(epoch)
        scheduler_d.step(epoch)

        print("Epoch : %d/%d" %(epoch, args.max_epochs) )
        print("Learning rate for G: %.9f" %lr_g)
        print("Learning rate for D: %.9f" %lr_d)

        # train update model
        Tr_losses = train(args, train_loader, 
                          (model_g, model_d, vocoder, model_STT,  decoder_STT),
                          (criterion_recon, criterion_ctc, criterion_adv, criterion_cl, CER), 
                          (optimizer_g, optimizer_d), 
                          epoch,
                          generation_counts,empty_counts,True)

        print('======================')
        print('Finish Training..............')

        # val 评估性能
        Val_losses = train(args, val_loader,
                           (model_g, model_d, vocoder, model_STT,  decoder_STT),
                           (criterion_recon, criterion_ctc, criterion_adv, criterion_cl, CER), 
                           ([],[]), 
                           epoch,
                            generation_counts,
                            empty_counts,
                           False,
                            )
        generation_counts = Val_losses[-2]
        empty_counts = Val_losses[-1]
        for cls in args.word_label:
            print(f"  {cls}:")
            print(f"    Empty counts: {empty_counts[cls]}")
            for generated_text, count in generation_counts[cls].items():
                print(f"    Generated '{generated_text}': {count} times")
        
        # Save checkpoint
        state_g = {'arch': str(model_g),
                 'state_dict': model_g.state_dict(),
                 'epoch': epoch,
                 'optimizer_state_dict': optimizer_g.state_dict()}
        
        state_d = {'arch': str(model_d),
                 'state_dict': model_d.state_dict(),
                 'epoch': epoch,
                 'optimizer_state_dict': optimizer_d.state_dict()}
        
        # Did validation loss improve?
        loss_total =  Val_losses[0]
        is_best = loss_total < best_loss
        best_loss = min(loss_total, best_loss)

        if not is_best:
            epochs_since_improvement += 1
            print("\nEpochs since last improvement: %d\n" % (epochs_since_improvement,))
        else:
            epochs_since_improvement = 0

        save_checkpoint(state_g, is_best, args.savemodel, 'checkpoint_g.pt')
        save_checkpoint(state_d, is_best, args.savemodel, 'checkpoint_d.pt')

        saveData(args, val_loader, (model_g, model_d, vocoder, model_STT, decoder_STT), epoch, (Tr_losses,Val_losses))

        time_taken = time.time() - start_time
        print("Time: %.2f\n"%time_taken)
        
    args.writer.flush()

if __name__ == '__main__':

    dataDir = './dataset'
    logDir = './TrainResult'
    
    parser = argparse.ArgumentParser(description='Hyperparams')
    parser.add_argument('--vocoder_pre', type=str, default='./pretrained_model/UNIVERSAL_V1/g_02500000', help='pretrained vocoder file path')
    parser.add_argument('--trained_model', type=str, default='./pretrained_model', help='trained model for G & D folder path')
    parser.add_argument('--model_config', type=str, default='./models', help='config for G & D folder path')
    parser.add_argument('--dataLoc', type=str, default=dataDir)
    parser.add_argument('--config', type=str, default='./config.json')
    parser.add_argument('--logDir', type=str, default=logDir)
    parser.add_argument('--resume', type=bool, default=False)  # 是否重新训练
    parser.add_argument('--pretrain', type=bool, default=False)  # 是否使用使用预训练模型
    parser.add_argument('--prefreeze', type=bool, default=False)  # 是否冻结预训练层
    parser.add_argument('--gpuNum', type=list, default=[0])
    parser.add_argument('--batch_size', type=int, default=26)
    parser.add_argument('--sub', type=str, default='sub1')  # 子目录名称
    parser.add_argument('--task', type=str, default='SpokenEEG')
    parser.add_argument('--recon', type=str, default='Y_mel')
    parser.add_argument('--unseen', type=str, default='stop')
    
    args = parser.parse_args()
    
    with open(args.config) as f:
        t_args = argparse.Namespace()
        t_args.__dict__.update(json.load(f))
        args = parser.parse_args(namespace=t_args)


    main(args)        
    
    
    
