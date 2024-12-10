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
from utils import Logger, set_seed, GenIdx
from data_loader import SYSUData, RegDBData, TestData, LLCMData
from data_manager import process_query_sysu, process_gallery_sysu, process_test_regdb, process_gallery_llcm, process_query_llcm
from model.network import embed_net
from engine import tester


def main_worker(args, args_main):
    # set gpu id and seed id
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    torch.backends.cudnn.benchmark = True  # accelerate the running speed of convolution network
    device = "cuda" if torch.cuda.is_available() else "cpu"
    set_seed(args.seed, cuda=torch.cuda.is_available())

    ## set file
    if not os.path.isdir(args.dataset + "_" + args.file_name):
        os.makedirs(args.dataset + "_" + args.file_name)
    file_name = args.dataset + "_" + args.file_name

    if args.dataset == "sysu":
        data_path = args.dataset_path + "SYSU-MM01/"
        log_path = os.path.join(file_name, args.dataset + "_" + args.log_path)
        test_mode = [1, 2]
    elif args.dataset == "regdb":
        data_path = args.dataset_path + "RegDB/"
        log_path = os.path.join(file_name, args.dataset + "_" + args.log_path)
        if args.mode == "thermaltovisible":
            test_mode = [1, 2]
        elif args.mode == "visibletothermal":
            test_mode = [2, 1]
    elif args.dataset == 'llcm':
        data_path = args.dataset_path + "LLCM/"
        log_path = os.path.join(file_name, args.dataset + "_" + args.log_path)
        test_mode = [2, 1]  # [2, 1]: VIS to IR; [1, 2]: IR to VIS

    if not os.path.isdir(log_path):
        os.makedirs(log_path)

    sys.stdout = Logger(os.path.join(log_path, "log_test.txt"))

    ## load data
    print("args_main:{}".format(args_main))
    print("args:{}".format(args))
    print("==> Loading data...")

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transform_test = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((args.img_h, args.img_w)),
        transforms.ToTensor(),
        normalize,
    ])

    end = time.time()

    if args.dataset == "sysu":
        # testing set
        query_img, query_label, query_cam = process_query_sysu(data_path, mode=args.mode)
        gall_img, gall_label, gall_cam = process_gallery_sysu(data_path, mode=args.mode)
    elif args.dataset == "regdb":
        # testing set
        query_img, query_label = process_test_regdb(data_path, trial=args.trial, modality=args.mode.split("to")[0])
        gall_img, gall_label = process_test_regdb(data_path, trial=args.trial, modality=args.mode.split("to")[1])
    elif args.dataset == "llcm":
        # testing set
        query_img, query_label, query_cam = process_query_llcm(data_path, mode=test_mode[1])
        gall_img, gall_label, gall_cam = process_gallery_llcm(data_path, mode=test_mode[0])

    print("Dataset {} Statistics:".format(args.dataset))
    print("  ----------------------------")
    print("  subset   | # ids | # images")
    print("  ----------------------------")
    print("  query    | {:5d} | {:8d}".format(len(np.unique(query_label)), len(query_label)))
    print("  gallery  | {:5d} | {:8d}".format(len(np.unique(gall_label)), len(gall_label)))
    print("  ----------------------------")
    #print("Data loading time:\t {:.3f}".format(time.time() - end))

    if args.dataset == "sysu":
        n_class = 395  # initial value
    elif args.dataset == "regdb":
        n_class = 206  # initial value
    elif args.dataset == "llcm":
        n_class = 713
    else:
        n_class = 1000  # initial value
    epoch = 0  # initial value

    if args_main.resume:
        resume_path = args_main.resume_path
        if os.path.isfile(resume_path):
            checkpoint = torch.load(resume_path)
            if "main_net" in checkpoint.keys():
                n_class = checkpoint["main_net"]["classifier.weight"].size(0)
            elif "net" in checkpoint.keys():
                n_class = checkpoint["net"]["classifier.weight"].size(0)
            epoch = checkpoint["epoch"]
            print("==> Loading checkpoint {} (epoch {}, number of classes {})".format(resume_path, epoch, n_class))
        else:
            print(
                "==> No checkpoint is found at {} (epoch {}, number of classes {})".format(resume_path, epoch, n_class))
    else:
        print("==> No checkpont is loaded (epoch {}, number of classes {})".format(epoch, n_class))

    ## build model
    main_net = embed_net(class_num=n_class, no_local=args.no_local, gm_pool=args.gm_pool, arch=args.arch,
                         share_net=args.share_net, pcb=args.pcb, local_feat_dim=256, num_strips=4)
    if args_main.resume and os.path.isfile(resume_path):
        if "main_net" in checkpoint.keys():
            main_net.load_state_dict(checkpoint["main_net"])
        elif "net" in checkpoint.keys():
            main_net.load_state_dict(checkpoint["net"])
    main_net.to(device)

    for trial in range(10):
        print('Test Trial: {}'.format(trial))
        if args.dataset == "sysu":
            # testing set
            query_img, query_label, query_cam = process_query_sysu(data_path, mode=args.mode)
            gall_img, gall_label, gall_cam = process_gallery_sysu(data_path, mode=args.mode, trial=trial)
        elif args.dataset == "regdb":
            # testing set
            test_trial = trial + 1
            query_img, query_label = process_test_regdb(data_path, trial=test_trial, modality=args.mode.split("to")[0])
            gall_img, gall_label = process_test_regdb(data_path, trial=test_trial, modality=args.mode.split("to")[1])
        elif args.dataset == "llcm":
            query_img, query_label, query_cam = process_query_llcm(data_path, mode=test_mode[1])
            gall_img, gall_label, gall_cam = process_gallery_llcm(data_path, mode=test_mode[0], trial=trial)

        gallset = TestData(gall_img, gall_label, transform_test=transform_test, img_size=(args.img_w, args.img_h))
        queryset = TestData(query_img, query_label, transform_test=transform_test, img_size=(args.img_w, args.img_h))

        # testing data loader
        gall_loader = data.DataLoader(gallset, batch_size=args.test_batch_size, shuffle=False, num_workers=args.workers)
        query_loader = data.DataLoader(queryset, batch_size=args.test_batch_size, shuffle=False,num_workers=args.workers)

        if args.dataset == "sysu":
            cmc_pool, mAP_pool, mINP_pool, cmc_fc, mAP_fc, mINP_fc = tester(args, epoch, main_net, test_mode, gall_label, gall_loader, query_label, query_loader,
                                    feat_dim=args.pool_dim, query_cam=query_cam, gall_cam=gall_cam)
        elif args.dataset == "regdb":
            cmc_pool, mAP_pool, mINP_pool, cmc_fc, mAP_fc, mINP_fc = tester(args, epoch, main_net, test_mode, gall_label, gall_loader, query_label, query_loader,
                                    feat_dim=args.pool_dim)
        elif args.dataset == "llcm":
            cmc_pool, mAP_pool, mINP_pool, cmc_fc, mAP_fc, mINP_fc = tester(args, epoch, main_net, test_mode, gall_label, gall_loader, query_label, query_loader,
                                    feat_dim=args.pool_dim, query_cam=query_cam,gall_cam=gall_cam)

        if trial == 0:
            all_cmc = cmc_fc
            all_mAP = mAP_fc
            all_mINP = mINP_fc
            all_cmc_pool = cmc_pool
            all_mAP_pool = mAP_pool
            all_mINP_pool = mINP_pool
        else:
            all_cmc = all_cmc + cmc_fc
            all_mAP = all_mAP + mAP_fc
            all_mINP = all_mINP + mINP_fc
            all_cmc_pool = all_cmc_pool + cmc_pool
            all_mAP_pool = all_mAP_pool + mAP_pool
            all_mINP_pool = all_mINP_pool + mINP_pool

        print('FC:Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
                cmc_fc[0], cmc_fc[4], cmc_fc[9], cmc_fc[19], mAP_fc, mINP_fc))
        print('POOL:Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
                cmc_pool[0], cmc_pool[4], cmc_pool[9], cmc_pool[19], mAP_pool, mINP_pool))

    cmc = all_cmc / 10
    mAP = all_mAP / 10
    mINP = all_mINP / 10

    cmc_pool = all_cmc_pool / 10
    mAP_pool = all_mAP_pool / 10
    mINP_pool = all_mINP_pool /10
    print('All Average:')
    print('FC: Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
                cmc[0], cmc[4], cmc[9], cmc[19], mAP, mINP))
    print('POOL: Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
            cmc_pool[0], cmc_pool[4], cmc_pool[9], cmc_pool[19], mAP_pool, mINP_pool))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OTLA-ReID for testing")
    parser.add_argument("--config", default="config/baseline.yaml", help="config file")
    parser.add_argument("--resume", action="store_true", help="resume from checkpoint")
    parser.add_argument("--resume_path", default="", help="checkpoint path")

    args_main = parser.parse_args()
    args = yaml.load(open(args_main.config), Loader=yaml.FullLoader)
    args = easydict.EasyDict(args)

    main_worker(args, args_main)














































