# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------
import math
import sys
from pathlib import Path
from typing import Iterable
from timm.utils import accuracy
import pandas as pd
import torch
import numpy as np
import util.misc as misc
import util.lr_sched as lr_sched
from sklearn.metrics import classification_report, confusion_matrix, multilabel_confusion_matrix, f1_score, accuracy_score, precision_score, recall_score
from models.evaluate import evaluate_internal


def train_one_epoch(model: torch.nn.Module, tokenizer,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, criterion, loss_scaler,
                    log_writer=None,
                    args=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 1

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    for data_iter_step, (img, txt) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
    # for data_iter_step, (img, txt) in enumerate(data_loader):

        # we use a per iteration (instead of per epoch) lr scheduler
        if args.scheduler:
            if data_iter_step % accum_iter == 0:
                lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)
        txt = list(txt)
        text_tokens = tokenizer(txt, return_tensors="pt", padding="max_length", truncation=True, max_length=512)

        img, text_tokens = img.to(device), text_tokens.to(device)
        # img, text_tokens = img.to(device, non_blocking=True), text_tokens.to(device, non_blocking=True)
        with torch.amp.autocast('cuda'):
            logits_img, logits_txt = model(img, text_tokens, device)
        loss = criterion(logits_img, logits_txt)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss /= accum_iter
        loss_scaler(loss, optimizer, parameters=model.parameters(),
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()
        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        # loss_value_reduce = misc.all_reduce_mean(loss_value)

        # if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
        #     """ We use epoch_1000x as the x-axis in tensorboard.
        #     This calibrates different curves when batch size changes.
        #     """
        #     epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
        #     log_writer.add_scalar('train_loss', loss_value_reduce, epoch_1000x)
        #     log_writer.add_scalar('lr', lr, epoch_1000x)


    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

@torch.no_grad()
def evaluate(epoch, data_loader, model, tokenizer, device, plot_dir):
    # model.module.text_transformer.resize_token_embeddings(len(tokenizer))
    softmax = torch.nn.Softmax(dim=0)
    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Test:'
    Path(plot_dir).mkdir(parents=True, exist_ok=True)
    # switch to evaluation mode
    model.eval()
    predictedall=[]
    realall=[]
    # # for _, (img, txt, onehotlabels, name_acc) in metric_logger.log_every(data_loader, 10, header):
    # for data_iter_step, (img, onehotlabels) in enumerate(metric_logger.log_every(data_loader, 10, header)):
    for data_iter_step, (nii_file, img, onehotlabels) in enumerate(data_loader):
        # text_tokens = tokenizer(txt, return_tensors="pt", padding="max_length", truncation=True, max_length=512)

        img = img.to(device)
        onehotlabels = onehotlabels.to(device)
        pathologies = ['Medical material',
                        'Arterial wall calcification', 
                        'Cardiomegaly', 
                        'Pericardial effusion',
                        'Coronary artery wall calcification', 
                        'Hiatal hernia',
                        'Lymphadenopathy', 
                        'Emphysema', 
                        'Atelectasis', 
                        'Lung nodule',
                        'Lung opacity', 
                        'Pulmonary fibrotic sequela', 
                        'Pleural effusion', 
                        'Mosaic attenuation pattern',
                        'Peribronchial thickening', 
                        'Consolidation', 
                        'Bronchiectasis',
                        'Interlobular septal thickening']  
# compute output

        with torch.amp.autocast('cuda'):
            predictedlabels=[]

            for pathology in pathologies:
                # text = [f"{pathology}.", f"not {pathology}."]   
                text = [f"{pathology} is present.", f"{pathology} is not present."]   

                text_tokens=tokenizer(
                                text, return_tensors="pt", padding="max_length", truncation=True, max_length=512).to(device)
                output = model(img, text_tokens,  device=device, validation=True)
                output = softmax(output)
                                
                append_out=output.detach().cpu().numpy()
                predictedlabels.append(append_out[0])

            predictedall.append(predictedlabels)
            realall.append(onehotlabels.detach().cpu().numpy()[0])
    realall=np.array(realall)
    predictedall=np.array(predictedall)

    dfs=evaluate_internal(predictedall,realall,pathologies, plot_dir)

    if epoch > 0:
        d = dfs.copy()
        dfr = [pd.read_excel(f'{plot_dir}aurocs_' + str(epoch - 1) + '.xlsx', sheet_name='Sheet' + str(i)) for i in range(len(dfs))]
        dfs = [pd.concat([dr, di], axis=0) for dr, di in zip(dfr, d)]
    
    writer = pd.ExcelWriter(f'{plot_dir}aurocs_' + str(epoch) + '.xlsx', engine='xlsxwriter')
    for i in range(len(dfs)):
        dfs[i].to_excel(writer, sheet_name='Sheet' + str(i), index=False)
    writer.close()

    realall = np.rint(realall).astype(int)
    predictedall = np.rint(predictedall).astype(int)
    f1 = f1_score(realall, predictedall,average='weighted')
    acc = accuracy_score(realall.flatten(), predictedall.flatten())
    pr = precision_score(realall, predictedall, average='micro')
    rec = recall_score(realall, predictedall,average='micro')
    print('Test F1 Score: ', f1)
    print('Test Accuracy: ', acc,'\n')
    print('Test Precision Score: ', pr)
    print('Test Recall: ', rec,'\n')

    metric_logger.meters['acc'].update(acc.item(), n=1)
    metric_logger.meters['f1'].update(f1.item(), n=1)
    metric_logger.meters['pr'].update(pr.item(), n=1)
    metric_logger.meters['rec'].update(rec.item(), n=1)
        # metric_logger.meters['f1'].update(f1.item(), n=1)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Acc {acc.global_avg:.3f} F1 {f1.global_avg:.3f} Prec {pr.global_avg:.3f} Rec {rec.global_avg:.3f}'
          .format(acc=metric_logger.acc, f1=metric_logger.f1, pr=metric_logger.pr, rec=metric_logger.rec))
    # print('* Acc {acc.global_avg:.3f} F1 {f1.global_avg:.3f}'
    #       .format(acc=metric_logger.acc, f1=metric_logger.f1))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}