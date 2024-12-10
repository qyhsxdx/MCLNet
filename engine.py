import time
import numpy as np
import torch
from torch.autograd import Variable
from utils import AverageMeter
from eval_metrics import eval_sysu, eval_regdb, eval_llcm
from re_rank import random_walk, k_reciprocal
from loss import KL_divergence


def trainer(args, epoch, main_net, adjust_learning_rate, optimizer, trainloader,
            criterion, writer = None, print_freq = 50):
    current_lr = adjust_learning_rate(args, optimizer, epoch)
    train_loss = AverageMeter()
    id_loss = AverageMeter()
    tri_loss = AverageMeter()
    mmd_loss = AverageMeter()
    data_time = AverageMeter()
    batch_time = AverageMeter()

    correct = 0
    total = 0

    main_net.train()  # switch to train model
    end = time.time()

    for batch_id, (input_rgb, input_ir, label_rgb, label_ir) in enumerate(trainloader):
        labels = torch.cat((label_rgb, label_ir), dim = 0)

        input1 = Variable(input_rgb.cuda())
        input2 = Variable(input_ir.cuda())
        labels = Variable(labels.cuda())
        data_time.update(time.time() - end)

        feat, out0 = main_net(input1, input2)
        loss_id = criterion[0](out0, labels)

        # loss_tri, batch_acc = criterion[1](feat[3], labels)
        # correct += (batch_acc / 2)
        # _, predicted = out0.max(1)
        # correct += (predicted.eq(labels).sum().item() / 2)
        # loss = loss_id + loss_tri * args.w_center
        loss = loss_id

        if args.dist_disc == "margin_mmd":
            feat_rgb, feat_ir = torch.split(feat[3], [label_rgb.size(0), label_ir.size(0)], dim = 0)
            loss_dist, l2max, expec = criterion[2](feat_rgb, feat_ir)
            loss = loss + loss_dist * args.dist_w

        for index in range(len(feat)):
            features = feat[index]
            loss = loss + criterion[3](features, labels) * args.supcon

        for index1 in range(len(feat)):
            o1 = feat[index1]
            for index2 in range(len(feat)):
                o2 = feat[index2]
                if o1 is not o2:
                    loss = loss + KL_divergence(o2.detach(), o1) * args.KL_div


        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        #update P
        train_loss.update(loss.item(), input1.size(0))
        id_loss.update(loss_id.item(), input1.size(0))
        #tri_loss.update(loss_tri, input1.size(0))
        total += labels.size(0)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if batch_id % print_freq == 0:
            print("Epoch:[{}][{}/{}]"
                  "Time:{batch_time.val:.3f}({batch_time.avg:.3f})"
                  "lr:{:.3f}"
                  "Loss:{train_loss.val:.4f}({train_loss.avg:.4f})"
                  "iLoss:{id_loss.val:.4f}({id_loss.avg:.4f})"
                  "TLoss:{tri_loss.val:.4f}({tri_loss.avg:.4f})"
                  "Accu:{:.2f}".format(
                epoch, batch_id, len(trainloader), current_lr,
                100. * correct / total, batch_time = batch_time,
                train_loss = train_loss, id_loss = id_loss, tri_loss = tri_loss))

        if writer is not None:
            writer.add_scalar("total_loss", train_loss.avg, epoch)
            writer.add_scalar("id_loss", id_loss.avg, epoch)
            writer.add_scalar("tri_loss", tri_loss.avg, epoch)
            writer.add_scalar("lr", current_lr, epoch)


def tester(args, epoch, main_net, test_mode, gall_label, gall_loader, query_label, query_loader,
           feat_dim = 2048, query_cam = None, gall_cam = None, writer = None):
    # switch to evaluation mode
    main_net.eval()
    print("Extracting Gallery Feature...")
    ngall = len(gall_label)
    start = time.time()
    ptr = 0
    feat_dim = feat_dim
    gall_feat = np.zeros((ngall, feat_dim))
    gall_feat_att = np.zeros((ngall, feat_dim))
    with torch.no_grad():
        for batch_idx, (input, label) in enumerate(gall_loader):
            batch_num = input.size(0)
            input = Variable(input.cuda())
            feat, feat_att = main_net(input, input, test_mode[0])
            feat = feat[3]
            gall_feat[ptr:ptr+batch_num, :] = feat.detach().cpu().numpy()
            gall_feat_att[ptr:ptr+batch_num, :] = feat_att.detach().cpu().numpy()
            ptr = ptr + batch_num
    print("Extracting Time:\t {:.3f}".format(time.time() - start))

    print("Extracting Query Feature...")
    nquery = len(query_label)
    start = time.time()
    ptr = 0
    query_feat = np.zeros((nquery, feat_dim))
    query_feat_att = np.zeros((nquery, feat_dim))
    with torch.no_grad():
        for batch_idx, (input, label) in enumerate(query_loader):
            batch_num = input.size(0)
            input = Variable(input.cuda())
            feat, feat_att = main_net(input, input, test_mode[1])
            feat = feat[3]
            query_feat[ptr:ptr+batch_num, :] = feat.detach().cpu().numpy()
            query_feat_att[ptr:ptr+batch_num, :] = feat_att.detach().cpu().numpy()
            ptr = ptr + batch_num
    print("Extracting Time:\t {:.3f}".format(time.time() - start))

    start = time.time()
    
    distmat = -np.matmul(query_feat, np.transpose(gall_feat))
    distmat_att = -np.matmul(query_feat_att, np.transpose(gall_feat_att))

    # evaluation
    if args.dataset == 'regdb':
        cmc, mAP, mINP = eval_regdb(distmat, query_label, gall_label)
        cmc_att, mAP_att, mINP_att = eval_regdb(distmat_att, query_label, gall_label)
    elif args.dataset == 'sysu':
        cmc, mAP, mINP = eval_sysu(distmat, query_label, gall_label, query_cam, gall_cam)
        cmc_att, mAP_att, mINP_att = eval_sysu(distmat_att, query_label, gall_label, query_cam, gall_cam)
    elif args.dataset == 'llcm':
        cmc, mAP, mINP = eval_llcm(distmat, query_label, gall_label, query_cam, gall_cam)
        cmc_att, mAP_att, mINP_att = eval_llcm(distmat_att, query_label, gall_label, query_cam, gall_cam)
    print('Evaluation Time:\t {:.3f}'.format(time.time() - start))

    if writer is not None:
        writer.add_scalar("rank1", cmc[0], epoch)
        writer.add_scalar("mAP", mAP, epoch)
        writer.add_scalar("mINP", mINP, epoch)
        writer.add_scalar('rank1_att', cmc_att[0], epoch)
        writer.add_scalar('mAP_att', mAP_att, epoch)
        writer.add_scalar('mINP_att', mINP_att, epoch)

    return cmc, mAP, mINP, cmc_att, mAP_att, mINP_att











































