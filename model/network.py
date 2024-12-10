import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
from .resnet import resnet50

class Normalize(nn.Module):
    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm)
        return out


class Non_local(nn.Module):
    def __init__(self, in_channels, reduc_ratio=2):
        super(Non_local, self).__init__()

        self.in_channels = in_channels
        self.inter_channels = reduc_ratio//reduc_ratio

        self.g = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1, stride=1,
                    padding=0),
        )

        self.W = nn.Sequential(
            nn.Conv2d(in_channels=self.inter_channels, out_channels=self.in_channels,
                    kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(self.in_channels),
        )
        nn.init.constant_(self.W[1].weight, 0.0)
        nn.init.constant_(self.W[1].bias, 0.0)



        self.theta = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                             kernel_size=1, stride=1, padding=0)

        self.phi = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                           kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        '''
                :param x: (b, c, t, h, w)
                :return:
                '''

        batch_size = x.size(0)
        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)
        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)
        f = torch.matmul(theta_x, phi_x)
        N = f.size(-1)
        # f_div_C = torch.nn.functional.softmax(f, dim=-1)
        f_div_C = f / N

        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        W_y = self.W(y)
        z = W_y + x

        return z

# #####################################################################
def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
        init.zeros_(m.bias.data)
    elif classname.find('BatchNorm1d') != -1:
        init.normal_(m.weight.data, 1.0, 0.01)
        init.zeros_(m.bias.data)

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        init.normal_(m.weight.data, 0, 0.001)
        if m.bias:
            init.zeros_(m.bias.data)

def gem(x):
    b, c, h, w = x.shape
    x = x.view(b, c, -1)
    p = 3.0
    x_pool = (torch.mean(x ** p, dim=1) + 1e-12) ** (1 / p)
    return x_pool


class visible_module(nn.Module):
    def __init__(self, arch = "resnet50", share_net=1):
        super(visible_module, self).__init__()

        model_v = resnet50(pretrained=True, last_conv_stride = 1, last_conv_dilation = 1)
        # avg pooling to global pooling
        self.share_net = share_net

        if self.share_net == 0:
            pass
        else:
            self.visible = nn.ModuleList()
            self.visible.conv1 = model_v.conv1
            self.visible.bn1 = model_v.bn1
            self.visible.relu = model_v.relu
            self.visible.maxpool = model_v.maxpool
            if self.share_net > 1:
                for i in range(1, self.share_net):
                    setattr(self.visible, "layer" + str(i), getattr(model_v, "layer" + str(i)))
                    setattr(self.visible, "auxiliary" + str(i), getattr(model_v, "auxiliary" + str(i)))

    def forward(self, x):
        if self.share_net == 0:
            return x
        else:
            x = self.visible.conv1(x)
            x = self.visible.bn1(x)
            x = self.visible.relu(x)
            x = self.visible.maxpool(x)

            if self.share_net > 1:
                feat_list = []
                for i in range(1, self.share_net):
                    x = getattr(self.visible, "layer" + str(i))(x)
                    out_feature = getattr(self.visible, "auxiliary" + str(i))(x)
                    feat_list.append(out_feature)

            return x, feat_list


class thermal_module(nn.Module):
    def __init__(self, arch = "resnet50", share_net=1):
        super(thermal_module, self).__init__()

        model_t = resnet50(pretrained=True, last_conv_stride = 1, last_conv_dilation = 1)
        # avg pooling to global pooling
        self.share_net = share_net

        if self.share_net == 0:
            pass
        else:
            self.thermal = nn.ModuleList()
            self.thermal.conv1 = model_t.conv1
            self.thermal.bn1 = model_t.bn1
            self.thermal.relu = model_t.relu
            self.thermal.maxpool = model_t.maxpool
            if self.share_net > 1:
                for i in range(1, self.share_net):
                    setattr(self.thermal, "layer" + str(i), getattr(model_t, "layer" + str(i)))
                    setattr(self.thermal, "auxiliary" + str(i), getattr(model_t, "auxiliary" + str(i)))

    def forward(self, x):
        if self.share_net == 0:
            return x
        else:
            x = self.thermal.conv1(x)
            x = self.thermal.bn1(x)
            x = self.thermal.relu(x)
            x = self.thermal.maxpool(x)

            if self.share_net > 1:
                feat_list = []
                for i in range(1, self.share_net):
                    x = getattr(self.thermal, "layer" + str(i))(x)
                    out_feature = getattr(self.thermal, "auxiliary" + str(i))(x)
                    feat_list.append(out_feature)
            return x, feat_list


class base_resnet(nn.Module):
    def __init__(self, arch = "resnet50", share_net = 1):
        super(base_resnet, self).__init__()

        model_base = resnet50(pretrained=True, last_conv_stride = 1, last_conv_dilation = 1)
        #avg pooling to global pooling
        model_base.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.share_net = share_net
        if self.share_net == 0:
            self.base = model_base
        else:
            self.base = nn.ModuleList()

            if self.share_net > 4:
                pass
            else:
                for i in range(self.share_net, 5):
                    setattr(self.base, "layer" + str(i), getattr(model_base, "layer" + str(i)))
                    setattr(self.base, "auxiliary" + str(3), getattr(model_base, "auxiliary" + str(3)))
    def forward(self, x):
        if self.share_net == 0:
            x = self.base.conv1(x)
            x = self.base.bn1(x)
            x = self.base.relu(x)
            x = self.base.maxpool(x)

            x = self.base.layer1(x)
            x = self.base.layer2(x)
            x = self.base.layer3(x)
            x = self.base.layer4(x)
        elif self.share_net > 4:
            return x
        else:
            feat_list = []
            for i in range(self.share_net, 5):
                x = getattr(self.base, "layer" + str(i))(x)
                if i == 3:
                    out_feature = getattr(self.base, "auxiliary" + str(3))(x)
                    feat_list.append(out_feature)
            return x, feat_list

class embed_net(nn.Module):
    def __init__(self, class_num, no_local = "off", gm_pool = "on", arch = "resnet50", share_net = 1, pcb = "off",
                 local_feat_dim = 256, num_strips = 4):
        super(embed_net, self).__init__()

        self.thermal_module = thermal_module(arch=arch, share_net=share_net)
        self.visible_module = visible_module(arch=arch, share_net=share_net)
        self.base_resnet = base_resnet(arch=arch, share_net=share_net)

        self.non_local = no_local
        self.pcb = pcb

        if self.non_local == "on":
            layers = [3, 4, 6, 3]
            non_layers = [0, 2, 3, 0]
            self.NL_1 = nn.ModuleList(
                [Non_local(256) for i in range(non_layers[0])])
            self.NL_1_idx = sorted([layers[0] - (i+1) for i in range(non_layers[0])])

            self.NL_2 = nn.ModuleList(
                [Non_local(512) for i in range(non_layers[1])])
            self.NL_2_idx = sorted([layers[1] - (i + 1) for i in range(non_layers[1])])

            self.NL_3 = nn.ModuleList(
                [Non_local(1024) for i in range(non_layers[2])])
            self.NL_3_idx = sorted([layers[2] - (i + 1) for i in range(non_layers[2])])

            self.NL_4 = nn.ModuleList(
                [Non_local(2048) for i in range(non_layers[3])])
            self.NL_4_idx = sorted([layers[3] - (i + 1) for i in range(non_layers[3])])

        pool_dim = 2048
        self.l2norm = Normalize(2)
        self.gm_pool = gm_pool

        self.bottleneck = nn.BatchNorm1d(pool_dim)
        self.bottleneck.bias.requires_grad_(False)#no shift

        self.classifier = nn.Linear(pool_dim, class_num, bias=False)

        self.bottleneck.apply(weights_init_kaiming)
        self.classifier.apply(weights_init_classifier)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x1, x2, modal = 0):
        if modal == 0:
            x1, feat_list1 = self.visible_module(x1)
            x2, feat_list2 = self.thermal_module(x2)
            x = torch.cat((x1, x2), dim=0)
            x_auxiliary1 = torch.cat((feat_list1[0], feat_list2[0]), dim=0)
            x_auxiliary2 = torch.cat((feat_list1[1], feat_list2[1]), dim=0)
        elif modal == 1:
            x, feat_list1 = self.visible_module(x1)
            x_auxiliary1 = feat_list1[0]
            x_auxiliary2 = feat_list1[1]

        elif modal == 2:
            x, feat_list2 = self.thermal_module(x2)
            x_auxiliary1 = feat_list2[0]
            x_auxiliary2 = feat_list2[1]

        # shared block
        if self.non_local == "on":
            NL1_counter = 0
            if len(self.NL_1_idx) == 0: self.NL_1_idx = [-1]
            for i in range(len(self.base_resnet.base.layer1)):
                x = self.base_resnet.base.layer1[i](x)
                if i == self.NL_1_idx[NL1_counter]:
                    _, C, H, W = x.shape
                    x = self.NL_1[NL1_counter](x)
                    NL1_counter += 1
            # layer2
            NL2_counter = 0
            if len(self.NL_2_idx) == 0: self.NL_2_idx = [-1]
            for i in range(len(self.base_resnet.base.layer2)):
                x = self.base_resnet.base.layer2[i](x)
                if i == self.NL_2_idx[NL2_counter]:
                    _, C, H, W = x.shape
                    x = self.NL_2[NL2_counter](x)
                    NL2_counter += 1
            # layer3
            NL3_counter = 0
            if len(self.NL_3_idx) == 0: self.NL_3_idx = [-1]
            for i in range(len(self.base_resnet.base.layer3)):
                x = self.base_resnet.base.layer3[i](x)
                if i == self.NL_3_idx[NL1_counter]:
                    _, C, H, W = x.shape
                    x = self.NL_3[NL3_counter](x)
                    NL3_counter += 1
            # layer4
            NL4_counter = 0
            if len(self.NL_4_idx) == 0: self.NL_4_idx = [-1]
            for i in range(len(self.base_resnet.base.layer4)):
                x = self.base_resnet.base.layer4[i](x)
                if i == self.NL_4_idx[NL4_counter]:
                    _, C, H, W = x.shape
                    x = self.NL_4[NL4_counter](x)
                    NL4_counter += 1
        else:
            x, feat_list3 = self.base_resnet(x)
            x_auxiliary3 = feat_list3[0]

        if self.gm_pool == "on":
            x_pool1 = gem(x_auxiliary1)
            x_pool2 = gem(x_auxiliary2)
            x_pool3 = gem(x_auxiliary3)
            x_pool4 = gem(x)
            x_pool = [x_pool1, x_pool2, x_pool3, x_pool4]
        else:
            x_pool1 = self.avgpool(x_auxiliary1)
            x_pool1 = x_pool1.view(x_pool1.size(0), x_pool1.size(1))
            x_pool2 = self.avgpool(x_auxiliary2)
            x_pool2 = x_pool2.view(x_pool2.size(0), x_pool2.size(1))
            x_pool3 = self.avgpool(x_auxiliary3)
            x_pool3 = x_pool3.view(x_pool3.size(0), x_pool3.size(1))
            x_pool4 = self.avgpool(x)
            x_pool4 = x_pool4.view(x_pool4.size(0), x_pool4.size(1))
            x_pool = [x_pool1, x_pool2, x_pool3, x_pool4]
        feat = self.bottleneck(x_pool4)

        if self.training:
            return x_pool, self.classifier(feat)
        else:
            x_pool_feat = []
            for index in range(len(x_pool)):
                x_pool_feature = self.l2norm(x_pool[index])
                x_pool_feat.append(x_pool_feature)
            return x_pool_feat, self.l2norm(feat)










































