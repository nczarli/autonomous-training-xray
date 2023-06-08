"""
**********************************************************************************
 * Autonomous Training in X-Ray Imaging Systems
 * 
 * Training a deep learning model based on noisy labels from a rule based algorithm.
 * 
 * Copyright 2023 Nikodem Czarlinski
 * 
 * Licensed under the Attribution-NonCommercial 3.0 Unported (CC BY-NC 3.0)
 * (the "License"); you may not use this file except in compliance with 
 * the License. You may obtain a copy of the License at
 * 
 *     https://creativecommons.org/licenses/by-nc/3.0/
 * 
**********************************************************************************
"""

import argparse

import torch.optim.lr_scheduler
import torchvision.transforms as transforms
import wandb
from torch.optim import SGD, Adam
from torch.utils.data import Subset
from tqdm import tqdm
import multiprocess as multiprocessing

from datasets.dataloader_imagefolder import cifar_dataset
from models.preresnet import PreResNet18, PreResNet34, PreResNet50, PreResNet101, PreResNet152
from utils import *
import dill as pickle
import time

parser = argparse.ArgumentParser('Train with synthetic cifar noisy dataset')
parser.add_argument('--dataset', default='cifar10', help='dataset name')


# model settings
parser.add_argument('--theta_s', default=1.0, type=float, help='threshold for selecting samples (default: 1)')
parser.add_argument('--theta_r', default=0.9, type=float, help='threshold for relabelling samples (default: 0.9)')
parser.add_argument('--lambda_fc', default=1.0, type=float, help='weight of feature consistency loss (default: 1.0)')
parser.add_argument('--k', default=200, type=int, help='neighbors for knn sample selection (default: 200)')

# train settings
parser.add_argument('--model', default='PreResNet18', help=f'model architecture (default: PreResNet18)')
parser.add_argument('--epochs', default=300, type=int, metavar='N', help='number of total epochs to run (default: 300)')
parser.add_argument('--batch_size', default=128, type=int, help='mini-batch size (default: 128)')
parser.add_argument('--lr', default=0.02, type=float, help='initial learning rate (default: 0.02)')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum of SGD solver (default: 0.9)')
parser.add_argument('--weight_decay', default=5e-4, type=float, help='weight decay (default: 5e-4)')
parser.add_argument('--seed', default=3047, type=int, help='seed for initializing training. (default: 3047)')
parser.add_argument('--gpuid', default='0', type=str, help='Selected GPU (default: "0")')
parser.add_argument('--entity', type=str, help='Wandb user entity')
parser.add_argument('--run_path', type=str, help='run path containing all results')
parser.add_argument('--use_cpu_only', default=False, type=bool ,help='use cpu only (default: False)')


def train(labeled_trainloader1,iter_labeled_loader2, modified_label, all_trainloader, encoder, classifier, proj_head, pred_head, optimizer, epoch, args):
    encoder.train()
    classifier.train()
    proj_head.train()
    pred_head.train()
    xlosses = AverageMeter('xloss')
    ulosses = AverageMeter('uloss')
    labeled_train_iter = labeled_trainloader1
    all_bar = tqdm(all_trainloader)

    for batch_idx, ([inputs_u1, inputs_u2], _, _) in enumerate(all_bar):
        try:
            [inputs_x1, inputs_x2], labels_x, index = next(labeled_train_iter)
        except:
            [inputs_x1, inputs_x2], labels_x, index = next(iter_labeled_loader2)

        # cross-entropy training with mixup
        batch_size = inputs_x1.size(0)
        inputs_x1, inputs_x2 = inputs_x1.to(device), inputs_x2.to(device)
        labels_x = modified_label[index]
        targets_x = torch.zeros(batch_size, args.num_classes, device=inputs_x1.device).scatter_(1, labels_x.view(-1, 1), 1)
        l = np.random.beta(4, 4)
        l = max(l, 1 - l)
        all_inputs_x = torch.cat([inputs_x1, inputs_x2], dim=0)
        all_targets_x = torch.cat([targets_x, targets_x], dim=0)
        idx = torch.randperm(all_inputs_x.size()[0])
        input_a, input_b = all_inputs_x, all_inputs_x[idx]
        target_a, target_b = all_targets_x, all_targets_x[idx]
        mixed_input = l * input_a + (1 - l) * input_b
        mixed_target = l * target_a + (1 - l) * target_b

        logits = classifier(encoder(mixed_input))
        Lce = -torch.mean(torch.sum(F.log_softmax(logits, dim=1) * mixed_target, dim=1))

        # optional feature-consistency
        inputs_u1, inputs_u2 = inputs_u1.to(device), inputs_u2.to(device)

        feats_u1 = encoder(inputs_u1)
        feats_u2 = encoder(inputs_u2)
        f, h = proj_head, pred_head

        z1, z2 = f(feats_u1), f(feats_u2)
        p1, p2 = h(z1), h(z2)
        Lfc = D(p2, z1)
        loss = Lce + args.lambda_fc * Lfc
        xlosses.update(Lce.item())
        ulosses.update(Lfc.item())
        # all_bar.set_description(
        #     f'Train epoch {epoch} LR:{optimizer.param_groups[0]["lr"]} Labeled loss: {xlosses.avg:.4f} Unlabeled loss: {ulosses.avg:.4f}')

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    logger.log({'ce loss': xlosses.avg, 'fc loss': ulosses.avg})




def test(testloader, encoder, classifier, epoch):
    encoder.eval()
    classifier.eval()
    accuracy = AverageMeter('accuracy')

    TP = 0
    TN = 0
    FP = 0
    FN = 0



    data_bar = tqdm(testloader)
    with torch.no_grad():
        for i, (data, label, _) in enumerate(data_bar):
            data, label = data.to(device), label.to(device)
            feat = encoder(data)
            res = classifier(feat)
            pred = torch.argmax(res, dim=1)

            # compute accuracy
            acc = torch.sum(pred == label) / float(data.size(0))
            accuracy.update(acc.item(), data.size(0))
            # data_bar.set_description(f'Test epoch {epoch}: Accuracy#{accuracy.avg:.4f}')

            TP += torch.sum((pred == 0) & (label == 0))
            TN += torch.sum((pred == 1) & (label == 1))
            FP += torch.sum((pred == 0) & (label == 1))
            FN += torch.sum((pred == 1) & (label == 0))

    # ensure no division by zero
    if TP + FP == 0:
        precision = 0
    else:
        precision = TP / (TP + FP)
    if TP + FN == 0:
        recall = 0
    else:
        recall = TP / (TP + FN)
    if precision + recall == 0:
        f1 = 0
    else:
        f1 = 2 * precision * recall / (precision + recall)

    accuracy_from_labels = (TP + TN) / (TP + TN + FP + FN)

    logger.log({'acc': accuracy.avg, 'precision': precision, 'recall': recall, 'f1': f1, 'TP': TP, 'TN': TN, 'FP': FP, 'FN': FN, 'acc_from_labels': accuracy_from_labels})
    return accuracy.avg


def evaluate(dataloader, encoder, classifier, args, noisy_label):
    """
    Evaluation function for strongly augmented data which is used for relabelling 
    and sample selection.
    """

    encoder.eval()
    classifier.eval()
    feature_bank = []
    prediction = []

    ################################### feature extraction ###################################
    with torch.no_grad():
        # generate feature bank
        for (data, target, index) in tqdm(dataloader, desc='Feature extracting'):
            data = data.to(device)
            feature = encoder(data)
            feature_bank.append(feature)
            res = classifier(feature)
            prediction.append(res)
        feature_bank = F.normalize(torch.cat(feature_bank, dim=0), dim=1)

        ################################### sample relabelling ###################################
        prediction_cls = torch.softmax(torch.cat(prediction, dim=0), dim=1)
        his_score, his_label = prediction_cls.max(1)
        print(f'Prediction track: mean: {his_score.mean()} max: {his_score.max()} min: {his_score.min()}')
        conf_id = torch.where(his_score > args.theta_r)[0]
        modified_label = torch.clone(noisy_label).detach()
        modified_label[conf_id] = his_label[conf_id]
        
        ################################### sample selection #####################################
        prediction_knn = weighted_knn(feature_bank, feature_bank, modified_label, args.num_classes, args.k, 10)  # temperature in weighted KNN
        vote_y = torch.gather(prediction_knn, 1, modified_label.view(-1, 1)).squeeze()
        vote_max = prediction_knn.max(dim=1)[0]
        right_score = vote_y / vote_max
        clean_id = torch.where(right_score >= args.theta_s)[0]
        noisy_id = torch.where(right_score < args.theta_s)[0]
    return clean_id, noisy_id, modified_label


def main():
    args = parser.parse_args()
    seed_everything(args.seed)
    if args.run_path is None:
        args.run_path = f'Dataset({args.dataset})_Runtime_Model(batch_size_{args.batch_size}_theta_r_{args.theta_r}theta_s_{args.theta_s}_lambda_fc_{args.lambda_fc}_k_{args.k}_)'

    global device
    # select device 
    if torch.cuda.is_available() and args.use_cpu_only == False:
        # Select the first available device
        device = torch.device("cuda:0")
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpuid
    else:
        # Use CPU if CUDA is not available
        device = torch.device("cpu")

    global logger
    logger = wandb.init(settings=wandb.Settings(start_method="thread"), project=args.dataset, entity=args.entity, name=args.run_path, group=args.dataset)
    logger.config.update(args)

    # create working directory
    if not os.path.isdir(f'working_dir/{args.dataset}'):
        os.mkdir(f'working_dir/{args.dataset}')
    if not os.path.isdir(f'working_dir/{args.dataset}/{args.run_path}'):
        os.mkdir(f'working_dir/{args.dataset}/{args.run_path}')

    ############################# Dataset initialization ##############################################

    args.num_classes = 2
    args.image_size = 32
    normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))


    # data loading
    weak_transform = transforms.Compose([
        transforms.RandomCrop(args.image_size, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize])
    none_transform = transforms.Compose([transforms.ToTensor(), normalize])
    strong_transform = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                           transforms.RandomHorizontalFlip(),
                                           CIFAR10Policy(),
                                           transforms.ToTensor(),
                                           normalize])
    
    train_path = 'C:\\Users\\Nikodem\\Documents\\GitHub\\autonomous-training-xray\\ssr\\datasets\\cifar-runtime-real\\train'
    val_path = 'C:\\Users\\Nikodem\\Documents\\GitHub\\autonomous-training-xray\\ssr\\datasets\\cifar-runtime-real\\val'

    # generate train dataset with only filtered clean subset
    train_data = cifar_dataset(dataset_dir=train_path, transform=KCropsTransform(strong_transform, 2),
                                 dataset_mode='train')
    eval_data = cifar_dataset(dataset_dir=train_path, transform=weak_transform,
                              dataset_mode='train')
    test_data = cifar_dataset( dataset_dir=val_path, transform=none_transform,
                              dataset_mode='test')
    all_data = cifar_dataset(dataset_dir=train_path,
                                   transform=MixTransform(strong_transform=strong_transform, weak_transform=weak_transform, K=1),
                                   dataset_mode='train')


    # extract noisy labels and clean labels for performance monitoring
    noisy_label = torch.tensor(eval_data.cifar_label).to(device)

    test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True)
    eval_loader = torch.utils.data.DataLoader(eval_data, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True)

    all_loader = torch.utils.data.DataLoader(all_data, batch_size=args.batch_size, num_workers=0, shuffle=True, pin_memory=True, drop_last=True)


    ################################ Model initialization ###########################################
    if args.model == 'PreResNet18':
        encoder = PreResNet18(args.num_classes)
    elif args.model == 'PreResNet34':
        encoder = PreResNet34(args.num_classes)
    elif args.model == 'PreResNet50':
        encoder = PreResNet50(args.num_classes)
    elif args.model == 'PreResNet101':
        encoder == PreResNet101(args.num_classes)
    elif args.model == 'PreResNet152':
        encoder = PreResNet152(args.num_classes)
    else:
        encoder = PreResNet18(args.num_classes)

    classifier = torch.nn.Linear(encoder.fc.in_features, args.num_classes)
    proj_head = torch.nn.Sequential(torch.nn.Linear(encoder.fc.in_features, 256),
                                    torch.nn.BatchNorm1d(256),
                                    torch.nn.ReLU(),
                                    torch.nn.Linear(256, 128))
    pred_head = torch.nn.Sequential(torch.nn.Linear(128, 256),
                                    torch.nn.BatchNorm1d(256),
                                    torch.nn.ReLU(),
                                    torch.nn.Linear(256, 128))
    encoder.fc = torch.nn.Identity()

    encoder.to(device)
    classifier.to(device)
    proj_head.to(device)
    pred_head.to(device)

    #################################### Training initialization #######################################
    optimizer = SGD([{'params': encoder.parameters()}, {'params': classifier.parameters()}, {'params': proj_head.parameters()}, {'params': pred_head.parameters()}],
                   lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum)
    
    # use adam instead of sgd
    # optimizer = torch.optim.Adam([{'params': encoder.parameters()}, {'params': classifier.parameters()}, {'params': proj_head.parameters()}, {'params': pred_head.parameters()}],)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr/50.0)

    acc_logs = open(f'working_dir/{args.dataset}/{args.run_path}/acc.txt', 'w')
    save_config(args, f'working_dir/{args.dataset}/{args.run_path}')
    print('Train args: \n', args)
    best_acc = 0

    ################################ Training loop ###########################################
    for i in range(args.epochs):
        clean_id, noisy_id, modified_label = evaluate(eval_loader, encoder, classifier, args, noisy_label)

        # balanced_sampler
        clean_subset = Subset(train_data, clean_id.cpu())
        sampler = ClassBalancedSampler(labels=modified_label[clean_id], num_classes=args.num_classes)

        # create labeled loader
        labeled_loader = torch.utils.data.DataLoader(clean_subset, batch_size=args.batch_size, sampler=sampler, num_workers=0,pin_memory=True, drop_last=True)
        
        # issues with finite loader, so we create 2 iterators
        iter_labeled_loader1 = iter(labeled_loader)
        iter_labeled_loader2 = iter(labeled_loader)

        train(iter_labeled_loader1, iter_labeled_loader2,  modified_label, all_loader, encoder, classifier, proj_head, pred_head, optimizer, i, args)

        cur_acc = test(test_loader, encoder, classifier, i)
        scheduler.step()
        if cur_acc > best_acc:
            best_acc = cur_acc
            save_checkpoint({
                'cur_epoch': i,
                'classifier': classifier.state_dict(),
                'encoder': encoder.state_dict(),
                'proj_head': proj_head.state_dict(),
                'pred_head': pred_head.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, filename=f'working_dir/{args.dataset}/{args.run_path}/best_acc.pth.tar')
        acc_logs.write(f'Epoch [{i}/{args.epochs}]: Best accuracy@{best_acc}! Current accuracy@{cur_acc} \n')
        acc_logs.flush()
        print(f'Epoch [{i}/{args.epochs}]: Best accuracy@{best_acc}! Current accuracy@{cur_acc} \n')

    save_checkpoint({
        'cur_epoch': args.epochs,
        'classifier': classifier.state_dict(),
        'encoder': encoder.state_dict(),
        'proj_head': proj_head.state_dict(),
        'pred_head': pred_head.state_dict(),
        'optimizer': optimizer.state_dict(),
    }, filename=f'working_dir/{args.dataset}/{args.run_path}/last1.pth.tar')


if __name__ == '__main__':
    # Print working directory
    print(os.getcwd())
    main()
