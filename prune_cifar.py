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
from Resnet_cifar import ResNet_cifar
import Eval_metric
import pdb

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


class net_pruner(object):
    def __init__(self, model, dataset,testset,model_type,cuda):
        self.model = model
        self.dataset = dataset
        self.testset = testset
        self.cuda = cuda
        self.model_type = model_type
        if int(self.model_type[3:])>50:
            self.use_bottleneck = True
        else:
            self.use_bottleneck = False
        # print(model)



    def evaluation_one_layer(self, layer_name, eval_metric, prune_ratio):
        """

        :param model: model to prune
        :param layer_name: layer to prune  format: layer1.n.conv1 n mean the index of bottleneck for resnet
        :param eval_metric:  eval metric
        :param prune_ratio: prune ratio
        :return:  index of filter to prune in this layer
        """
        model = self.model
        def get_sorted_index_by_grad_and_feature(eval_layer,eval_mode,model):
            batch_results=[]

            def feature_hook(module, input, output):
                eval_feature.append(output.data)

            def grad_hook(module, grad_input, grad_output):
                register_grad = grad_output[0]
                assert len(register_grad.size()) == 4, '梯度维度有误'
                eval_feature_grad.append(register_grad.data)

            handle_feat = eval_layer.register_forward_hook(feature_hook)
            handle_grad = eval_layer.register_backward_hook(grad_hook)
            optimizer = optim.SGD(params=model.parameters(), lr=0.1)
            model.eval()
            model = model.cuda()

            for data, target in self.dataset:
                eval_feature = []
                eval_feature_grad = []
                optimizer.zero_grad()
                data, target = data.float().cuda(), target.long().cuda()
                output = model(data)
                loss = F.cross_entropy(output, target)
                loss.backward()
                batch_feature = eval_feature[-1]
                batch_grad = eval_feature_grad[-1]
                if eval_mode == 'kernel_similarity':
                    weight_grad = eval_layer.weight.grad
                    O, C, H, W = weight_grad.size()
                    weight_grad = weight_grad.view(O, C * H * W)
                    norm_grad = torch.norm(weight_grad, dim=1).view(O, 1)
                    weight_grad = weight_grad / norm_grad
                    T_weight_grad = weight_grad.transpose(1, 0)
                    weight_grad_matrix = torch.einsum('jk,kl->jl', (weight_grad, T_weight_grad)).cpu().numpy()

                    ##这是一个i,j相关度的矩阵,问题：要找到和其他线性度最高的是不是应该要绝对值？？
                    weight_grad_matrix = np.abs(weight_grad_matrix)
                    #####################################################

                    kernel_grad_raw_sum = np.sum(weight_grad_matrix, axis=1)
                    batch_results.append(kernel_grad_raw_sum)
            handle_feat.remove()
            handle_grad.remove()
            # pdb.set_trace()
            # print(len(batch_results))
            all_results = np.average(batch_results,axis=0)
            assert len(all_results) == O , 'result计算结果维度不匹配'
            sorted_index = np.argsort(all_results)
            return  sorted_index





        def get_feature_and_grad(eval_layer, model):
            eval_feature = []
            eval_feature_grad = []

            def feature_hook(module, input, output):
                eval_feature.append(output.data)

            def grad_hook(module, grad_input, grad_output):
                register_grad = grad_output[0]
                assert len(register_grad.size()) == 4, '梯度维度有误'
                eval_feature_grad.append(register_grad.data)

            handle_feat = eval_layer.register_forward_hook(feature_hook)
            handle_grad = eval_layer.register_backward_hook(grad_hook)
            optimizer = optim.SGD(params=model.parameters(),lr=0.1)

            optimizer.zero_grad()
            for data, target in self.dataset:

                model.eval()
                model = model.cuda()
                data, target = data.float().cuda(), target.long().cuda()
                output = model(data)
                loss = F.cross_entropy(output, target)
                loss.backward()


            handle_feat.remove()
            handle_grad.remove()
            eval_feature = eval_feature[-1]
            eval_feature_grad = eval_feature_grad[-1]
            # print(eval_feature_grad)
            return eval_feature, eval_feature_grad

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

            eval_feature,eval_feature_grad = get_feature_and_grad(eval_layer,model)
            sort_by_l2 = np.argsort(np.sqrt(np.sum(eval_feature * eval_feature, axis=(0, 2, 3))), axis=0)
            layer_index = sort_by_l2[:prune_num]
            return layer_index
        if eval_metric == 'Taylor':
            print('prune by taylor ')
            eval_feature, eval_feature_grad = get_feature_and_grad(eval_layer, model)
            taylor_value = np.abs(np.average(eval_feature*eval_feature_grad,axis=(0,2,3)))
            taylor_sort = np.argsort(taylor_value,axis=0)
            prun_index = taylor_sort[:prune_num]
            return prun_index
        if eval_metric == 'feature_grim':
            eval_feature,eval_feature_grad = get_feature_and_grad(eval_layer,model)
            grim_maxtix_avg = Eval_metric.feature_grim_matrix(eval_feature)
            np.set_printoptions(threshold=np.NAN)
            connection_sort_arg =np.flip(np.argsort(np.sum(np.abs(grim_maxtix_avg),axis=1)),axis=0)
            prun_index = connection_sort_arg[:prune_num]
            return prun_index
        if eval_metric == 'kernel_similarity':
            print('prunning by kernel grad similarity')
            sorted_by_kernel_similarity = get_sorted_index_by_grad_and_feature(eval_layer,'kernel_similarity',model)
            layer_index = sorted_by_kernel_similarity[(-1*prune_num):]
            # layer_index = sorted_by_grad_similarity[:prune_num]
            return layer_index




    def prun_entire_filter_by_index(self,layer_name,pruning_index):
        #这里剪枝的时候还需要把BN的值给保留下来
        eval_layer = self.get_layer_by_name(layer_name)
        old_weight = eval_layer.weight.data
        chanel_num = old_weight.size(0)
        prune_num = len(pruning_index)
        new_chanel_num = chanel_num - len(pruning_index)
        chanel_weight = torch.split(old_weight, 1, 0)
        new_weight = [chanel_weight[i] for i in range(chanel_num) if i not in pruning_index]
        new_weight = torch.cat(new_weight, dim=0)
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
        # import pdb
        # pdb.set_trace()
        name_list = layer_name.split('.')
        l_index = name_list[0]
        b_index = int(name_list[1])
        c_index = name_list[2]
        bn_name = l_index+'.'+str(b_index)+'.'+'bn'+c_index[-1]
        old_bn = self.get_layer_by_name(bn_name)
        new_bn = self.reset_bn_after_conv(old_bn,pruning_index)
        # new_bn.num_batches_tracked = bn_num_batches_tracked
        model_layer = getattr(self.model, l_index)
        bottle_neck = model_layer[b_index]
        setattr(bottle_neck, 'bn'+c_index[-1], new_bn)



        next_layer_name = self.set_layer(layer_name, new_layer)
        return next_layer_name

    def prun_filter_chanel_by_index(self,layer_name,pruning_index):
        next_layer_name = layer_name
        next_layer = self.get_layer_by_name(next_layer_name)
        old_weight = next_layer.weight.data
        chanel_num = old_weight.size(1)
        prune_num = len(pruning_index)
        old_filter_list = torch.split(old_weight, 1, dim=0)
        new_filter_list = []
        for single_filter in old_filter_list:
            p_filter = torch.stack([single_filter[0][i] for i in range(chanel_num) if i not in pruning_index], dim=0)
            new_filter_list.append(p_filter)
        new_weight = torch.stack(new_filter_list, dim=0)
        new_next_layer = torch.nn.Conv2d(in_channels=next_layer.in_channels - prune_num,
                                         out_channels=next_layer.out_channels,
                                         kernel_size=next_layer.kernel_size,
                                         stride=next_layer.stride,
                                         padding=next_layer.padding,
                                         bias=next_layer.bias,
                                         dilation=next_layer.dilation,
                                         groups=next_layer.groups)

        new_next_layer.weight.data = new_weight
        _ = self.set_layer(next_layer_name, new_next_layer)

    def pruning_normal_layer(self,layer_name,pruning_index):
        """


        :param layer_name:  layer to prune
        :param pruning_index:  filter index in layer to prune
        :return: prune model
        """
        ## res18 没有conv3. 所以要考虑一下conv2连接shortcut的该怎么剪枝
        #  要根据模型类别 判断这一层是不是连接着shortcut 如果连接着shortcut，就判断是不是downsample如果是downsample再考虑剪枝，不然不减，写一个get_affected layer 的函数

        next_layer_name = self.prun_entire_filter_by_index(layer_name,pruning_index)
        self.prun_filter_chanel_by_index(next_layer_name,pruning_index)




        # print(self.model)

    def prun_last_conv_layer(self,layer_name,pruning_index):
        name_list = layer_name.split('.')
        l_index = name_list[0]
        b_index = int(name_list[1])
        c_index = name_list[2]
        if b_index == 0 and l_index != 'layer1':

            downsample_name = l_index+'.'+str(b_index)+'.'+'downsample'
            downsample = self.get_layer_by_name(downsample_name)
            eval_layer = downsample[0]
            old_weight = eval_layer.weight.data
            chanel_num = old_weight.size(0)
            prune_num = len(pruning_index)
            new_chanel_num = chanel_num - len(pruning_index)
            chanel_weight = torch.split(old_weight, 1, 0)
            new_weight = [chanel_weight[i] for i in range(chanel_num) if i not in pruning_index]
            new_weight = torch.cat(new_weight, dim=0)
            new_layer = torch.nn.Conv2d(in_channels=eval_layer.in_channels,
                                        out_channels=new_chanel_num,
                                        kernel_size=eval_layer.kernel_size,
                                        stride=eval_layer.stride,
                                        padding=eval_layer.padding,
                                        bias=eval_layer.bias,
                                        dilation=eval_layer.dilation,
                                        groups=eval_layer.groups)

            new_layer.weight.data = new_weight

            ##reset bn after conv
            old_bn = downsample[1]
            new_bn = self.reset_bn_after_conv(old_bn,pruning_index)
            ##end
            new_downsample = nn.Sequential(
                new_layer,
                new_bn
            )
            res_layer = getattr(self.model,l_index)
            block = res_layer[b_index]
            setattr(block,'downsample',new_downsample)

            affected_last_conv = layer_name
            _ = self.prun_entire_filter_by_index(affected_last_conv,pruning_index)


            #所有的bottleneck的conv1和last_conv都要处理
            for i in range(1,len(res_layer)):
                affected_first_conv = l_index+'.'+str(i)+'.'+'conv1'
                self.prun_filter_chanel_by_index(affected_first_conv,pruning_index)
                affected_last_conv = l_index+'.'+str(i)+'.'+c_index
                _ = self.prun_entire_filter_by_index(affected_last_conv,pruning_index)
            if not l_index == 'layer4':
                l_index = 'layer'+str(int(l_index[-1])+1)
                affected_first_conv = l_index+'.'+'0'+'.'+'conv1'
                self.prun_filter_chanel_by_index(affected_first_conv, pruning_index)

                ## affected downsample
                downsample_name =  l_index + '.' + '0' + '.' + 'downsample'
                downsample = self.get_layer_by_name(downsample_name)
                eval_layer = downsample[0]

                old_weight = eval_layer.weight.data
                chanel_num = old_weight.size(1)
                prune_num = len(pruning_index)
                old_filter_list = torch.split(old_weight, 1, dim=0)
                new_filter_list = []
                for single_filter in old_filter_list:
                    p_filter = torch.stack([single_filter[0][i] for i in range(chanel_num) if i not in pruning_index],
                                           dim=0)
                    new_filter_list.append(p_filter)
                new_weight = torch.stack(new_filter_list, dim=0)
                new_next_layer = torch.nn.Conv2d(in_channels=eval_layer.in_channels - prune_num,
                                                 out_channels=eval_layer.out_channels,
                                                 kernel_size=eval_layer.kernel_size,
                                                 stride=eval_layer.stride,
                                                 padding=eval_layer.padding,
                                                 bias=eval_layer.bias,
                                                 dilation=eval_layer.dilation,
                                                 groups=eval_layer.groups)

                new_next_layer.weight.data = new_weight
                new_downsample = nn.Sequential(
                    new_next_layer,
                    downsample[1]
                )
                res_layer = getattr(self.model, l_index)
                block = res_layer[b_index]
                setattr(block, 'downsample', new_downsample)
                ##end
            else:
                fc =self.model.fc
                fc_weight = fc.weight.data
                fc_bias = fc.bias.data
                fc_weight_list = torch.split(fc_weight,1,1)
                # print(fc_weight_list[1].size())
                new_fc_weight = torch.cat([fc_weight_list[i] for i in range(chanel_num) if i not in pruning_index],dim=1)
                num_class = fc.out_features
                new_fc = nn.Linear(new_chanel_num,num_class)
                new_fc.weight.data = new_fc_weight
                new_fc.bias.data =fc_bias
                self.model.fc =new_fc
        else:
            pass








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
        optimizer = optim.SGD(params=self.model.parameters(), lr=1e-3, momentum=0.9, weight_decay=0.0001)
        best_acc = 0
        best_model = self.model
        for epoch in range(40):
            test_acc = test_cifar(self.model, epoch,self.testset)
            train_cifar(self.model, epoch, optimizer)
            if test_acc > best_acc:
                best_acc = test_acc
                best_model = self.model
                print('best acc is ', best_acc)
        test_acc = test_cifar(self.model, epoch, self.testset)
        torch.save(best_model, save_path)


    def pruning_process(self,pruning_ratio,finetune_method,eval_metric,save_path):
        """

        :param pruning_ratio: pruning ratio for every layer
        :param finetune_method: 1.finetune after  all layers were pruned 2.fine tune with re_init model
        :return:
        """
        model =self.model
        use_bottleneck= self.use_bottleneck
        def prepare_all_layers_for_pruning(model):

            layers_for_pruning = []
            if isinstance(model,ResNet_cifar):
                print('pruning model of Resnet type')
                # layers = ['layer1', 'layer2', 'layer3', 'layer4']
                layers = ['layer1']
                if use_bottleneck:

                    for layer in layers:
                        res_layer = getattr(model, layer)
                        len_bottle = len(res_layer)
                        for index in range(len_bottle):
                            conv1_name = layer + '.' + str(index) + '.' + 'conv1'
                            conv2_name = layer + '.' + str(index) + '.' + 'conv2'
                            conv3_name = layer + '.' + str(index) + '.' + 'conv3'
                            layers_for_pruning.append(conv1_name)
                            layers_for_pruning.append(conv2_name)
                            layers_for_pruning.append(conv3_name)
                else:

                    for layer in layers:
                        res_layer = getattr(model, layer)
                        len_bottle = len(res_layer)
                        for index in range(len_bottle):
                            conv1_name = layer + '.' + str(index) + '.' + 'conv1'
                            conv2_name = layer + '.' + str(index) + '.' + 'conv2'
                            layers_for_pruning.append(conv1_name)
                            layers_for_pruning.append(conv2_name)



            print(layers_for_pruning)
            return layers_for_pruning

        layers_for_pruning = prepare_all_layers_for_pruning(model)
        if self.use_bottleneck:
            last_conv_name = 'conv3'
        else:
            last_conv_name = 'conv2'

        ##test code                                  问题在这
        # layers_for_pruning = ['layer3.0.conv1', 'layer3.0.conv2','layer3.1.conv1']
        # layer4 剪枝后的精度会有波动 layer2没有。。还有layer2为什么精度下降这么多
        #end
        if finetune_method == 1:
            for layer_to_prun in layers_for_pruning:
                if layer_to_prun.split('.')[-1] == last_conv_name :
                    pruning_filter_index = self.evaluation_one_layer(layer_to_prun, eval_metric, pruning_ratio)
                    self.prun_last_conv_layer(layer_to_prun,pruning_filter_index)
                else:
                    pruning_filter_index = self.evaluation_one_layer(layer_to_prun, eval_metric, pruning_ratio)
                    self.pruning_normal_layer(layer_to_prun,pruning_filter_index)
            print(self.model)
            self.finetune_model('cifar', save_path)
        elif finetune_method == 2:
            for layer_to_prun in layers_for_pruning:
                if layer_to_prun.split('.')[-1] == last_conv_name :
                    pruning_filter_index = self.evaluation_one_layer(layer_to_prun, eval_metric, pruning_ratio)
                    self.prun_last_conv_layer(layer_to_prun,pruning_filter_index)
                else:
                    pruning_filter_index = self.evaluation_one_layer(layer_to_prun, eval_metric, pruning_ratio)
                    self.pruning_normal_layer(layer_to_prun,pruning_filter_index)
            print(self.model)
            self.re_init_model()
            self.train_scratch('cifar', save_path)





    def train_scratch(self,train_data,save_path):
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
        optimizer = optim.SGD(params=self.model.parameters(), lr=1e-1, momentum=0.9, weight_decay=0.0001)
        best_acc = 0
        best_model = self.model
        for epoch in range(200):
            if epoch > 80:
                base_lr = 1e-2
                if epoch > 100:
                    base_lr = 1e-3
                if epoch > 120:
                    base_lr = 1e-4
                if epoch > 140:
                    base_lr = 1e-5
                optimizer.param_groups[0]['lr'] = base_lr

            test_acc = test_cifar(self.model, epoch,self.testset)
            train_cifar(self.model, epoch, optimizer)
            if test_acc > best_acc:
                best_acc = test_acc
                best_model = self.model
                print('best acc is ', best_acc)
        test_acc = test_cifar(self.model, epoch, self.testset)
        torch.save(best_model, save_path)




    def set_layer(self,layer_name,new_layer):
        name_list = layer_name.split('.')
        l_index = name_list[0]
        b_index = int(name_list[1])
        c_index = name_list[2]
        model_layer = getattr(self.model, l_index)
        bottle_neck = model_layer[b_index]
        # new_bn = nn.BatchNorm2d(new_layer.out_channels)
        # new_bn_name = 'bn'+c_index[-1]
        setattr(bottle_neck, c_index, new_layer)
        # setattr(bottle_neck,new_bn_name,new_bn)

        if c_index[-1] == '1':
            next_layer = l_index+'.'+ name_list[1] + '.conv2'
        elif c_index[-1] == '2':
            next_layer =  l_index+'.'+name_list[1]+'.conv3'
        else :
            next_layer = ''
        return next_layer


    def get_layer_by_name(self,layer_name):
        # print('search',layer_name)

        model = self.model
        for name, sub_module in model.named_modules():
            if name == layer_name:
                eval_layer = sub_module
        return eval_layer


    def reset_bn_after_conv(self,old_bn,pruning_index):
        chanel_num = old_bn.weight.size(0)
        new_chanel_num = chanel_num - len(pruning_index)
        bn_running_mean = old_bn.running_mean.data
        bn_running_var = old_bn.running_var
        bn_weight = old_bn.weight
        bn_bias = old_bn.bias
        # print(bn_running_mean)
        # bn_num_batches_tracked = old_bn.num_batches_tracked
        new_running_mean = [bn_running_mean[i]  for i in range(chanel_num) if i not in pruning_index ]
        new_running_mean = torch.Tensor(new_running_mean)
        new_running_var = [bn_running_var[i]  for i in range(chanel_num) if i not in pruning_index ]
        new_running_var = torch.Tensor(new_running_var)
        new_bn_weight = [bn_weight[i] for i in range(chanel_num) if i not in pruning_index]
        new_bn_weight = torch.Tensor(new_bn_weight)
        new_bn_bias = [bn_bias[i] for i in range(chanel_num) if i not in pruning_index]
        new_bn_bias = torch.Tensor(new_bn_bias)
        new_bn = nn.BatchNorm2d(new_chanel_num)
        new_bn.running_mean.data = new_running_mean
        new_bn.running_var.data = new_running_var
        new_bn.weight.data = new_bn_weight
        new_bn.bias.data = new_bn_bias
        return new_bn
    def re_init_model(self):
        for child_module in self.model.modules():
            if isinstance(child_module,nn.Conv2d) or isinstance(child_module,nn.BatchNorm2d) or isinstance(child_module,nn.Linear):
                child_module.reset_parameters()


##从python2.7产生的model文件需要在unpick的时候加上encoding = ‘utf-8'
model = torch.load('../model/resnet18_kernel3_best.th')
pruner = net_pruner(model, data_train,data_test,model_type='res18',cuda=True)
# p_index = pruner.evaluation_one_layer(layer_name='layer1.0.conv1',eval_metric='feature_grim',prune_ratio=0.3)
# pruner.pruning_normal_layer('layer1.0.conv1',p_index)
# pruner.re_init_model()
# pruner.finetune_model('cifar','./')
pruner.pruning_process(pruning_ratio=0.3,finetune_method=1,eval_metric='kernel_similarity',save_path='../model/pruned_model/resnet18_93.9_layer1_0.3_kernel_similarity.th')

