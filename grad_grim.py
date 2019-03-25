import torch
import torchvision.models as models
import torch.nn as nn
from torchvision import datasets
from torchvision import transforms
# from torch_affine_conv.layers import ConvAffine2d
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
import numpy as np
import pickle
import Resnet_cifar
import Eval_metric
import pdb
from tensorboardX import SummaryWriter

def test_cifar(model, epoch):
    model.eval()
    test_loss = 0
    correct = 0
    for tub in data_test:
        data, target = tub
        data, target = data.float().cuda(), target.long().cuda()
        # print(data.data)
        data, target = Variable(data), Variable(target)
        output = model(data)
        test_loss += F.cross_entropy(output, target).data[0]
        lable = output.data.max(1)[1]
        correct += lable.eq(target.data).cpu().sum()
    test_loss /= len(data_test)  # loss function already averages over batch size
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, 10000, 100. * correct / 10000))
    return float(correct) / 10000



def train_grim_matrix(model,epoch,optimizer):
    global step

    def feature_hook(module, input, output):
        # eval_feature.append(output.data.cpu().numpy())
        eval_feature.append(output.data)

    def grad_hook(module, grad_input, grad_output):
        # eval_feature_grad.append(grad_output[0].data.cpu().numpy())

        eval_feature_grad.append(grad_output.data)

    def clear_handle(handle_list):
        for t in handle_list:
            t.remove()
    model.train()
    correct=0
    handle_feat_list = []
    handle_grad_list = []
    layer_name_list = []
    ## 添加hook
    for name, child_module in model.named_modules():
        if isinstance(child_module, nn.Conv2d):
            layer_name_list.append(name)
            handle_feat = child_module.register_forward_hook(feature_hook)
            handle_feat_list.append(handle_feat)
            handle_grad = child_module.register_forward_hook(grad_hook)
            handle_grad_list.append(handle_grad)
    ##一个batch
    for tub in data_train:
        data, target = tub
        data, target = data.float().cuda(), target.long().cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        eval_feature = []
        eval_feature_grad = []
        feature_grad_matrix_list = []
        kernel_grad_matrix_list = []

        # pdb.set_trace()
        output = model(data)
        lable = output.data.max(1)[1]
        correct += lable.eq(target.data).cpu().sum()
        loss = F.cross_entropy(output, target)
        loss.backward()

        if  step%100 == 0:
            ##feature grad grim matrix
            # for index in range(len(eval_feature)):
            #     # pdb.set_trace()
            #
            #     grad = eval_feature_grad[index]
            #     B, C, H, W = grad.size()
            #     grad = grad.view(B, C, H * W)
            #     ##梯度图归一化
            #     norm_grad = torch.norm(grad,dim=2).view(B,C,1)
            #     grad = grad/norm_grad
            #     T_grad = grad.transpose(2, 1)
            #     grad_matrix_batch = torch.einsum('ijk,ikl->ijl',(grad,T_grad))
            #     grad_matrix_avg = (torch.sum(grad_matrix_batch,dim=0)/B).cpu().numpy()
            #     feature_grad_matrix_list.append(grad_matrix_avg)
            #     if index == 0 :
            #         feat_grad_raw_sum = np.sum(grad_matrix_avg,axis=1)
            #         writer.add_histogram('feat_grad_histogram',feat_grad_raw_sum,step)
            #         for i in range(len(feat_grad_raw_sum)):
            #             writer.add_scalar('feat_grad_'+str(i),feat_grad_raw_sum[i],step)
            #         writer.add_image('feat_grad_matrix',grad_matrix_avg,step)


            ## kernel grad grim matrix
            ## 不归一化的分布也可以看一下
            for name, child_module in model.named_modules():
                if isinstance(child_module, nn.Conv2d):
                    # pdb.set_trace()
                    weight_grad = child_module.weight.grad
                    O,C, H, W = weight_grad.size()
                    weight_grad = weight_grad.view(O,C* H * W)
                    ##梯度图归一化
                    norm_grad = torch.norm(weight_grad, dim=1).view(O, 1)
                    weight_grad = weight_grad / norm_grad
                    T_weight_grad = weight_grad.transpose(1, 0)
                    weight_grad_matrix = torch.einsum('jk,kl->jl', (weight_grad, T_weight_grad)).cpu().numpy()
                    # grad_matrix_avg = (torch.sum(weight_grad, dim=0) / B).cpu().numpy()
                    kernel_grad_matrix_list.append(weight_grad_matrix)
                    if name in ['conv1','layer4.0.conv1']:
                        for registe_type in ['raw_sum','one_by_one']:
                            if registe_type == 'raw_sum':
                                kernel_grad_raw_sum = np.sum(weight_grad_matrix,axis=1)
                                writer.add_histogram(name+registe_type, kernel_grad_raw_sum, step)
                                for i in range(len(kernel_grad_raw_sum)):
                                    writer.add_scalar('kernel_grad' + str(i), kernel_grad_raw_sum[i], step)
                            if registe_type == 'one_by_one':
                                kernel_grad_one_by_one = np.reshape(weight_grad_matrix,(-1,1))
                                writer.add_histogram(name+registe_type,kernel_grad_one_by_one,step)
                if isinstance(child_module,nn.Linear)






        optimizer.step()
        step=step+1
    clear_handle(handle_feat_list)
    clear_handle(handle_grad_list)


transform_train = transforms.Compose(
    [transforms.Pad(3, 1), transforms.RandomCrop(32),
     transforms.RandomHorizontalFlip(), transforms.ToTensor(),
     transforms.Normalize(mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225])])
transform_test = transforms.Compose([ transforms.ToTensor(),
                                     transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                          std=[0.229, 0.224, 0.225])])
cifar_train = datasets.CIFAR10('../cifar10', train=True, download=True, transform=transform_train)
cifar_test = datasets.CIFAR10('../cifar10', train=False, download=True, transform=transform_test)
data_train = torch.utils.data.DataLoader(cifar_train, batch_size=256, shuffle=True,num_workers=16)
data_test = torch.utils.data.DataLoader(cifar_test, batch_size=256, shuffle=True,num_workers=16)

resnet = Resnet_cifar.resnet18(pretrained=False)
resnet.fc = nn.Linear(in_features=512, out_features=10)
writer = SummaryWriter()


resnet = resnet.cuda()
resnet_param = resnet.parameters()
base_lr = 0.1
optimizer = optim.SGD(params=resnet.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
best_acc = 0
best_model = resnet

step=0
for epoch in range(160):
    if epoch > 80:
        base_lr = 1e-2
        if epoch > 100:
            base_lr = 1e-3
        if epoch > 120:
            base_lr = 1e-4
        if epoch > 140:
            base_lr = 1e-5
        optimizer.param_groups[0]['lr'] = base_lr
    # if epoch % 50 == 0:
    #     torch.save(resnet, 'model/resnet18_32*32_' + str(epoch) + '.th')
    train_grim_matrix(resnet, epoch,optimizer)
    test_acc = test_cifar(resnet, epoch)
    if test_acc > best_acc:
        best_acc = test_acc
        best_model = resnet
        print('best acc is ',best_acc)
writer.close()
torch.save(best_model, '../model/resnet18_grad_grim_matrix_1.th')