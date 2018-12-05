# encoding: utf-8
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

transform_train = transforms.Compose(
    [transforms.Pad(3, 1), transforms.RandomCrop(32), transforms.Resize(224),
     transforms.RandomHorizontalFlip(), transforms.ToTensor(),
     transforms.Normalize(mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225])])
transform_test = transforms.Compose([transforms.Resize(224), transforms.ToTensor(),
                                     transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                          std=[0.229, 0.224, 0.225])])
cifar_train = datasets.CIFAR10('../cifar10', train=True, download=True, transform=transform_train)
cifar_test = datasets.CIFAR10('../cifar10', train=False, download=True, transform=transform_test)
data_train = torch.utils.data.DataLoader(cifar_train, batch_size=32, shuffle=True,num_workers=16)
data_test = torch.utils.data.DataLoader(cifar_test, batch_size=32, shuffle=True,num_workers=16)


class net_pruner(object):
    def __init__(self, model, dataset,testset,cuda):
        self.model = model
        self.dataset = dataset
        self.testset = testset
        self.cuda = cuda



    def evaluation_one_layer(self, layer_name, eval_metric, prune_ratio):
        """

        :param model: model to prune
        :param layer_name: layer to prune  format: layer1.n.conv1 n mean the index of bottleneck for resnet
        :param eval_metric:  eval metric
        :param prune_ratio: prune ratio
        :return:  index of filter to prune in this layer
        """
        model = self.model
        if isinstance(model, models.ResNet):
            eval_layer = self.get_layer_by_name(layer_name)

            # Resnet conv1 [layer1,layer2,layer3,layer4],fc
            eval_weight = eval_layer.weight.data.cpu().numpy()
            num_filter = eval_weight.shape[0]
            prune_num = int(num_filter * prune_ratio)


            ## choose eval metric
            if eval_metric == 'L2':
                sort_by_l2 = np.argsort(np.sqrt(np.sum(eval_weight * eval_weight, axis=(1, 2, 3))), axis=0)
                layer_index = sort_by_l2[:prune_num]
                return layer_index
            if eval_metric == 'L1':
                sort_by_l1 = np.argsort(np.sum(np.abs(eval_weight), axis=(1, 2, 3)), axis=0)
                layer_index = sort_by_l1[:prune_num]
                return layer_index
            if eval_metric == 'activation':
                def get_feature_and_grad(eval_layer,model):
                    eval_feature = []
                    eval_feature_grad =[]
                    def feature_hook(module, input, output):

                        eval_feature.append(output.data.cpu().numpy())

                    def grad_hook(module,grad_input,grad_output):
                        print(grad_output[0].data.cpu().numpy())
                        eval_feature_grad.append(grad_output[0].data.cpu().numpy())


                    handle_feat = eval_layer.register_forward_hook(feature_hook)
                    handle_grad = eval_layer.register_backward_hook(grad_hook)
                    for data, target in self.dataset:
                        model.train()
                        model = model.cuda()
                        data,target = data.float().cuda(),target.long().cuda()
                        output = model(data)
                        loss = F.cross_entropy(output, target)
                        loss.backward()

                        break
                    handle_feat.remove()
                    handle_grad.remove()
                    eval_feature = eval_feature[0]
                    eval_feature_grad = eval_feature_grad[0]
                    # print(eval_feature_grad)
                    return eval_feature,eval_feature_grad
                eval_feature,eval_feature_grad = get_feature_and_grad(eval_layer,model)
                sort_by_l2 = np.argsort(np.sqrt(np.sum(eval_feature * eval_feature, axis=(0, 2, 3))), axis=0)
                layer_index = sort_by_l2[:prune_num]
                return layer_index




    def pruning_layer_by_Index(self,layer_name,pruning_index):
        """


        :param layer_name:  layer to prune
        :param pruning_index:  filter index in layer to prune
        :return: prune model
        """
        eval_layer = self.get_layer_by_name(layer_name)
        old_weight = eval_layer.weight.data
        chanel_num = old_weight.size(0)
        prune_num = len(pruning_index)
        new_chanel_num = chanel_num-len(pruning_index)
        chanel_weight = torch.split(old_weight,1,0)
        new_weight = [chanel_weight[i] for i in range(chanel_num) if i not in pruning_index]
        new_weight = torch.cat(new_weight,dim=0)
        new_layer = torch.nn.Conv2d(in_channels=eval_layer.in_channels,
                                    out_channels=new_chanel_num,
                                    kernel_size=eval_layer.kernel_size,
                                    stride=eval_layer.stride,
                                    padding=eval_layer.padding,
                                    bias=eval_layer.bias,
                                    dilation=eval_layer.dilation,
                                    groups=eval_layer.groups)


        new_layer.weight.data = new_weight
        # only support for layer[n].n.conv1 or conv2

        next_layer_name = self.set_layer(layer_name,new_layer)
        next_layer = self.get_layer_by_name(next_layer_name)
        old_weight = next_layer.weight.data
        old_filter_list = torch.split(old_weight,1,dim=0)
        new_filter_list = []
        for single_filter in old_filter_list:
            p_filter = torch.stack([single_filter[0][i] for i in range(chanel_num) if i not in pruning_index],dim=0)
            new_filter_list.append(p_filter)
        new_weight = torch.stack(new_filter_list,dim=0)
        new_next_layer = torch.nn.Conv2d(in_channels=next_layer.in_channels-prune_num,
                                    out_channels=next_layer.out_channels,
                                    kernel_size=next_layer.kernel_size,
                                    stride=next_layer.stride,
                                    padding=next_layer.padding,
                                    bias=next_layer.bias,
                                    dilation=next_layer.dilation,
                                    groups=next_layer.groups)

        new_next_layer.weight.data = new_weight
        _ = self.set_layer(next_layer_name,new_next_layer)
        # print(self.model)


    def finetune_model(self,train_data,save_path):
        """

        :param train_data: dataset used while training the model
        :param save_path: path to save the pruned model
        :return:
        """
        cuda = self.cuda
        def train_cifar(model, epoch,optimizer):
            if cuda:
                model =model.cuda()


            model.train()
            correct = 0
            for tub in self.dataset:
                data, target = tub
                data, target = data.float().cuda(), target.long().cuda()
                data, target = Variable(data), Variable(target)
                optimizer.zero_grad()
                output = model(data)
                lable = output.data.max(1)[1]
                correct += lable.eq(target.data).cpu().sum()
                loss = F.cross_entropy(output, target)
                loss.backward()
                optimizer.step()
            print('Train Epoch: {}\tLoss: {:.6f},train_acc:{:.2f}'.format(epoch, loss.data[0], float(correct) / 50000))

        def test_cifar(model, epoch,testset):
            if cuda:
                model = model.cuda()

            model.eval()
            test_loss = 0
            correct = 0
            for tub in testset:
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

        # if train_data == 'cifar'
        optimizer = optim.SGD(params=self.model.parameters(), lr=1e-3, momentum=0.9, weight_decay=0.0005)
        best_acc = 0
        best_model = self.model
        for epoch in range(32):
            if epoch > 16:
                base_lr = 1e-4
                if epoch > 20:
                    base_lr = 1e-5
                optimizer.param_groups[0]['lr'] = base_lr
            train_cifar(self.model,epoch,optimizer)
            test_acc = test_cifar(self.model, epoch,self.testset)
            if test_acc > best_acc:
                best_acc = test_acc
                best_model = self.model
                print('best acc is ', best_acc)
        torch.save(best_model, save_path)


    def pruning_process(self,pruning_ratio,finetune_method,eval_metric,save_path):
        """

        :param pruning_ratio: pruning ratio for every layer
        :param finetune_method: 1.finetune after  all layers were pruned
        :return:
        """
        model =self.model
        def prepare_all_layers_for_pruning(model):
            layers_for_pruning = []
            if isinstance(model,models.ResNet):
                print('pruning model of Resnet type')
                layers = ['layer1','layer2','layer3','layer4']
                for layer in layers:
                    res_layer = getattr(model,layer)
                    len_bottle = len(res_layer)
                    for index in range(len_bottle):
                        conv1_name = layer+'.'+str(index)+'.'+'conv1'
                        conv2_name = layer+'.'+str(index)+'.'+'conv2'
                        layers_for_pruning.append(conv1_name)
                        layers_for_pruning.append(conv2_name)



            return layers_for_pruning

        layers_for_pruning = prepare_all_layers_for_pruning(model)
        if finetune_method == 1:
            for prunning_layer in layers_for_pruning:
                print('pruning for {}'.format(prunning_layer))
                pruning_filter_index = self.evaluation_one_layer(prunning_layer,eval_metric,pruning_ratio)
                self.pruning_layer_by_Index(prunning_layer,pruning_filter_index)
            self.finetune_model('cifar',save_path)









    def set_layer(self,layer_name,new_layer):
        name_list = layer_name.split('.')
        l_index = name_list[0]
        b_index = int(name_list[1])
        c_index = name_list[2]
        model_layer = getattr(self.model, l_index)
        bottle_neck = model_layer[b_index]
        new_bn = nn.BatchNorm2d(new_layer.out_channels)
        new_bn_name = 'bn'+c_index[-1]
        setattr(bottle_neck, c_index, new_layer)
        setattr(bottle_neck,new_bn_name,new_bn)

        if c_index[-1] == '1':
            next_layer = l_index+'.'+ name_list[1] + '.conv2'
        elif c_index[-1] == '2':
            next_layer =  l_index+'.'+name_list[1]+'.conv3'
        else :
            next_layer = ''
        return next_layer


    def get_layer_by_name(self,layer_name):
        model = self.model
        for name, sub_module in model.named_modules():
            if name == layer_name:
                eval_layer = sub_module
        return eval_layer




##从python2.7产生的model文件需要在unpick的时候加上encoding = ‘utf-8'
model = torch.load('../model/resnet101_pretrained_fina_97.38%l.th')
pruner = net_pruner(model, data_train,data_test,cuda=True)
pruner.pruning_process(0.3,1,eval_metric='L2',save_path='../pruned_model/resnet101_97.38%_0.3.th')
