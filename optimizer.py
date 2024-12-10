import torch.optim as optim


def adjust_learning_rate(args, optimizer, epoch):
    if epoch < 10:
        lr = args.lr * (epoch + 1) / 10
    elif epoch >= 10 and epoch < 20:
        lr = args.lr
    elif epoch >= 20 and epoch < 50:
        lr = args.lr * 0.1
    elif epoch >= 50:
        lr = args.lr * 0.01

    optimizer.param_groups[0]["lr"] = 0.1 * lr
    for i in range(len(optimizer.param_groups) - 1):
        optimizer.param_groups[i + 1]["lr"] = lr

    return lr


def select_optimizer(args, main_net):
    if args.optim == "adam":
        ignored_params = list(map(id, main_net.bottleneck.parameters())) \
                         + list(map(id, main_net.classifier.parameters()))

        base_params = filter(lambda p: id(p) not in ignored_params, main_net.parameters())
        optimizer = optim.Adam([
            {"params": base_params, "lr": 0.1 * args.lr},
            {"params": main_net.bottleneck.parameters(), "lr": args.lr},
            {"params": main_net.classifier.parameters(), "lr": args.lr}],
            weight_decay=5e-4)
    elif args.optim == "sgd":
        ignored_params = list(map(id, main_net.bottleneck.parameters())) \
                         + list(map(id, main_net.classifier.parameters()))

        base_params = filter(lambda p: id(p) not in ignored_params, main_net.parameters())

        optimizer = optim.SGD([
            {'params': base_params, 'lr': 0.1 * args.lr},
            {'params': main_net.bottleneck.parameters(), 'lr': args.lr},
            {'params': main_net.classifier.parameters(), 'lr': args.lr}],
            weight_decay=5e-4, momentum=0.9, nesterov=True)

    return optimizer
