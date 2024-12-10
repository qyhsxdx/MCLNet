import argparse
import easydict
import yaml
import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter
from utils import Logger, set_seed, GenIdx, IdentitySampler
from data_loader import SYSUData, RegDBData, TestData, LLCMData
from data_manager import process_query_sysu, process_gallery_sysu, process_test_regdb, process_query_llcm, process_gallery_llcm
from model.network import embed_net
from loss import TripletLoss_WRT, MarginMMD_Loss, CenterTripletLoss, SupConLoss
from optimizer import select_optimizer, adjust_learning_rate
from engine import trainer, tester

def main_worker(args, args_main):
    # set start epoch and end epoch
    start_epoch = args.start_epoch
    end_epoch = args.end_epoch

    # set gpu id and seed id
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    torch.backends.cudnn.benchmark = True  # accelerate the running speed of convolution network
    device = "cuda" if torch.cuda.is_available() else "cpu"
    set_seed(args.seed, cuda=torch.cuda.is_available())

    # set file
    if not os.path.isdir(args.dataset + "_" + args.file_name):
        os.makedirs(args.dataset + "_" + args.file_name)
    file_name = args.dataset + "_" + args.file_name  # log file name

    if args.dataset == "sysu":
        data_path = args.dataset_path + "SYSU-MM01/"
        log_path = os.path.join(file_name, args.dataset + "_" + args.log_path)#log path
        vis_log_path = os.path.join(file_name, args.dataset + "_" + args.vis_log_path)#tensorboard log save path
        model_path = os.path.join(file_name, args.dataset + "_" + args.model_path)
        test_mode = [1, 2]
    elif args.dataset == "regdb":
        data_path = args.dataset_path + "RegDB/"
        log_path = os.path.join(file_name, args.dataset + "_" + args.log_path)  # log path
        vis_log_path = os.path.join(file_name, args.dataset + "_" + args.vis_log_path)  # tensorboard log save path
        model_path = os.path.join(file_name, args.dataset + "_" + args.model_path)
        if args.mode == "thermaltovisible":
            test_mode = [1, 2]
        elif args.mode == "visibletothermal":
            test_mode = [2, 1]
    elif args.dataset == "llcm":
        data_path = args.dataset_path + "LLCM/"
        log_path = os.path.join(file_name, args.dataset + "_" + args.log_path)  # log path
        vis_log_path = os.path.join(file_name, args.dataset + "_" + args.vis_log_path)  # tensorboard log save path
        model_path = os.path.join(file_name, args.dataset + "_" + args.model_path)
        test_mode = [2, 1]  # [1, 2]: IR to VIS; [2, 1]: VIS to IR;


    if not os.path.isdir(log_path):
        os.makedirs(log_path)
    if not os.path.isdir(vis_log_path):
        os.makedirs(vis_log_path)
    if not os.path.isdir(model_path):
        os.makedirs(model_path)

    sys.stdout = Logger(os.path.join(log_path, "log_os.txt"))
    test_os_log = open(os.path.join(log_path, "log_os_test.txt"), "w")

    # tensorboard
    writer = SummaryWriter(vis_log_path)

    # load data
    print("args_main:{}".format(args_main))
    print("args:{}".format(args))
    print("==>Loading data...")

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transform_train_rgb = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Pad(10),
        transforms.RandomGrayscale(p=0.5),
        transforms.RandomCrop((args.img_h, args.img_w)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
        transforms.RandomErasing(p=0.5),
    ])
    transform_train_ir = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Pad(10),
        transforms.RandomCrop((args.img_h, args.img_w)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
        transforms.RandomErasing(p=0.5),
    ])
    transform_test = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((args.img_h, args.img_w)),
        transforms.ToTensor(),
        normalize,
    ])

    end = time.time()
    if args.dataset == "sysu":
        # training set
        trainset = SYSUData(args, data_path, transform_train_rgb=transform_train_rgb,
                            transform_train_ir=transform_train_ir)
        # generate the idx of each person identity
        color_pos, thermal_pos = GenIdx(trainset.train_color_label, trainset.train_thermal_label)
        # testing set
        query_img, query_label, query_cam = process_query_sysu(data_path, mode=args.mode)
        gall_img, gall_label, gall_cam = process_gallery_sysu(data_path, mode=args.mode)
    elif args.dataset == "regdb":
        # training set
        trainset = RegDBData(args, data_path, transform_train_rgb=transform_train_rgb,
                             transform_train_ir=transform_train_ir)
        # generate the idx of each perosn identity
        color_pos, thermal_pos = GenIdx(trainset.train_color_label, trainset.train_thermal_label)
        # testing set
        query_img, query_label = process_test_regdb(data_path, trial=args.trial, modality=args.mode.split("to")[0])
        gall_img, gall_label = process_test_regdb(data_path, trial=args.trial, modality=args.mode.split("to")[1])
    elif args.dataset == 'llcm':
        # training set
        trainset = LLCMData(data_path, trial=args.trial, transform=transform_train_rgb)
        # generate the idx of each person identity
        color_pos, thermal_pos = GenIdx(trainset.train_color_label, trainset.train_thermal_label)

        # testing set
        query_img, query_label, query_cam = process_query_llcm(data_path, mode=test_mode[1])
        gall_img, gall_label, gall_cam = process_gallery_llcm(data_path, mode=test_mode[0])

    gallset = TestData(gall_img, gall_label, transform_test=transform_test, img_size=(args.img_w, args.img_h))
    queryset = TestData(query_img, query_label, transform_test=transform_test, img_size=(args.img_w, args.img_h))

    # testing data loader
    gall_loader = data.DataLoader(gallset, batch_size=args.test_batch_size, shuffle=False, num_workers=args.workers)
    query_loader = data.DataLoader(queryset, batch_size=args.test_batch_size, shuffle=False, num_workers=args.workers)

    n_class = len(np.unique(trainset.train_color_label))  # number of classes
    n_rgb = len(trainset.train_color_label)  # number of visible images
    n_ir = len(trainset.train_thermal_label)  # number of infrared images

    print("Dataset {} Statistics:".format(args.dataset))
    print("----------------------")
    print(" subset |# ids | # images")
    print(" visible |{:5d}|{:8d}".format(n_class, len(trainset.train_color_label)))
    print(" thermal |{:5d}|{:8d}".format(n_class, len(trainset.train_thermal_label)))
    print("----------------------")
    print(" query |{:5d}|{:8d}".format(len(np.unique(query_label)), len(query_label)))
    print(" gallery|{:5d}|{:8d}".format(len(np.unique(gall_label)), len(gall_label)))
    print("----------------------")
    print("Data loading time:\t{:.3f}".format(time.time() - end))

    print("==> Building model..")
    main_net = embed_net(n_class, no_local=args.no_local, gm_pool=args.gm_pool, arch=args.arch, share_net=args.share_net,
                         pcb=args.pcb, local_feat_dim=256, num_strips=4)
    main_net.to(device)

    # resume checkpoints
    if args_main.resume:
        resume_path = args_main.resume_path
        if os.path.isfile(resume_path):
            checkpoint = torch.load(resume_path)
            if "epoch" in checkpoint.keys():
                start_epoch = checkpoint["epoch"]
            main_net.load_state_dict(checkpoint["main_net"])
            print("==>Loading checkpoint {} (epoch {})".format(resume_path, start_epoch))
        else:
            print("==>No checkpoint if found at {}".format(resume_path))
    print("Start epoch:{}, end epoch:{}".format(start_epoch, end_epoch))

    # define loss functions
    criterion = []
    criterion_id = nn.CrossEntropyLoss()  # ID Loss
    criterion.append(criterion_id)
    criterion_tri = CenterTripletLoss()  # HC-Triplet loss
    criterion.append(criterion_tri)
    criterion_MarginMMD = MarginMMD_Loss() #MMD
    criterion.append(criterion_MarginMMD)
    criterion_SupCon = SupConLoss() #Supervision Contrastive loss
    criterion.append(criterion_SupCon)

    # set optimizer
    optimizer = select_optimizer(args, main_net)

    # start training and testing
    print("==> Start training...")
    best_acc = 0
    all_end = time.time()
    for epoch in range(start_epoch, end_epoch - start_epoch):
        end = time.time()
        print("==> Preparing data loader...")
        # identity sampler
        sampler = IdentitySampler(trainset.train_color_label, trainset.train_thermal_label, color_pos, thermal_pos,
                                  args.num_pos, args.train_batch_size, args.dataset_num_size)
        trainset.cIndex = sampler.index1  # color index
        trainset.tIndex = sampler.index2  # thermal index
        trainloader = data.DataLoader(trainset, batch_size=args.train_batch_size * args.num_pos, sampler=sampler,
                                      num_workers=args.workers, drop_last=True)

        # training
        trainer(args, epoch, main_net, adjust_learning_rate, optimizer, trainloader, criterion, writer)
        print("Training time per epoch:{:.3f}".format(time.time() - end))

        if epoch % args.eval_epoch == 0:
            if args.dataset == "sysu":
                print("Testing Epoch:{}, Testing mode:{}".format(epoch, args.mode))
                print("Testing Epoch:{}, Testing mode:{}".format(epoch, args.mode), file=test_os_log)
            elif args.dataset == "regdb":
                print("Testing Epoch:{}, Testing mode:{}, Trial:{}".format(epoch, args.mode, args.trial))
                print("Testing Epoch:{}, Testing mode:{}, Trial:{}".format(epoch, args.mode, args.trial),
                      file=test_os_log)
            elif args.dataset == "llcm":
                print("Testing Epoch:{}, Testing mode:{}".format(epoch, test_mode))
                print("Testing Epoch:{}, Testing mode:{}".format(epoch, test_mode), file=test_os_log)

            # start testing
            end = time.time()
            if args.dataset == "sysu":
                cmc, mAP, mINP, cmc_fc, mAP_fc, mINP_fc = tester(args, epoch, main_net, test_mode, gall_label, gall_loader, query_label,
                                        query_loader, feat_dim=args.pool_dim, query_cam=query_cam, gall_cam=gall_cam, writer=writer)
            elif args.dataset == "regdb":
                cmc, mAP, mINP, cmc_fc, mAP_fc, mINP_fc = tester(args, epoch, main_net, test_mode, gall_label, gall_loader, query_label, query_loader,
                                        feat_dim=args.pool_dim, writer=writer)
            elif args.dataset == "llcm":
                cmc, mAP, mINP, cmc_fc, mAP_fc, mINP_fc = tester(args, epoch, main_net, test_mode, gall_label, gall_loader, query_label,
                                        query_loader, feat_dim=args.pool_dim, query_cam=query_cam, gall_cam=gall_cam, writer=writer)
            print("Testing time per epoch:{:.3f}".format(time.time() - end))

            # save model
            if cmc_fc[0] > best_acc:
                best_acc = cmc_fc[0]
                best_epoch = epoch
                best_mAP = mAP_fc
                best_mINP = mINP_fc
                state = {
                    "main_net": main_net.state_dict(),
                    "cmc": cmc_fc,
                    "mAP": mAP_fc,
                    "mINP": mINP_fc,
                    "epoch": epoch,
                    "n_class": n_class,
                }
                torch.save(state, os.path.join(model_path, "best_checkpoint.pth"))

            if epoch % args.save_epoch == 0:
                state = {
                    "main_net": main_net.state_dict(),
                    "cmc": cmc_fc,
                    "mAP": mAP_fc,
                    "mINP": mINP_fc,
                    "epoch": epoch,
                    "n_class": n_class,
                }
                torch.save(state, os.path.join(model_path, "checkpoint_epoch{}.pth".format(epoch)))
            print('POOL: Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
                        cmc[0], cmc[4], cmc[9], cmc[19], mAP, mINP))
            print('FC: Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
                    cmc_fc[0], cmc_fc[4], cmc_fc[9], cmc_fc[19], mAP_fc, mINP_fc))
            print('Best Epoch [{}], Rank-1: {:.2%} |  mAP: {:.2%}| mINP: {:.2%}'.format(best_epoch, best_acc, best_mAP, best_mINP))
    print("Training time of overall:{:.3f}".format(time.time() - all_end))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VI-ReID")
    parser.add_argument("--config", default="config/baseline.yaml", help="config file")
    parser.add_argument("--resume", action="store_true", help="resume from checkpoint")
    parser.add_argument("--resume_path", default="", help="checkpoint path")

    args_main = parser.parse_args()
    args = yaml.load(open(args_main.config, "r"), Loader=yaml.FullLoader)
    args = easydict.EasyDict(args)

    main_worker(args, args_main)