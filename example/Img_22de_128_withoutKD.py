from __future__ import print_function, absolute_import
import sys
sys.path.append('./')
import argparse
import time
import matplotlib.pyplot as plt

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torchvision.datasets as datasets

from pose.utils.config import config
from pose.utils.logger import Logger, savefig
from pose.utils.evaluation import accuracy, AverageMeter, final_preds,cal_nme
from pose.utils.misc import save_checkpoint, save_pred, adjust_learning_rate
from pose.utils.osutils import mkdir_p, isfile, isdir, join
from pose.utils.imutils import batch_with_heatmap
from pose.utils.transforms import fliplr, flip_back
import pose.models as models
import pose.datasets as datasets
import pose.utils.criterion as criterion_pair

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

idx = [1]

best_nme = 1  # best_acc = 0

def parse():
    parser = argparse.ArgumentParser(description='PyTorch MobileFAN Training')
    # Model structure
    parser.add_argument('--num-classes', default=29, type=int, metavar='N',
                        help='Number of keypoints')
    # Training strategy
    parser.add_argument('-t_pixel', '--Temperature-pixel', default=1, type=float, metavar='N',
                        help='temperature')
    parser.add_argument('-t_pair', '--Temperature-pair', default=1, type=float, metavar='N',
                        help='temperature')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--epochs', default=80, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--train-batch', default=8, type=int, metavar='N',
                        help='train batchsize')
    parser.add_argument('--test-batch', default=8, type=int, metavar='N',
                        help='test batchsize')
    parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--momentum', default=0, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--weight-decay', '--wd', default=0, type=float,
                        metavar='W', help='weight decay (default: 0)')
    parser.add_argument('--schedule', type=int, nargs='+', default=[],
                        help='Decrease learning rate at these epochs.')
    parser.add_argument('--gamma', type=float, default=0.1,
                        help='LR is multiplied by gamma on schedule.')
    # Data processing
    parser.add_argument('-f', '--flip', dest='flip', action='store_true',
                        help='flip the input during validation')
    parser.add_argument('--sigma', type=float, default=1,
                        help='Groundtruth Gaussian sigma.')
    parser.add_argument('--sigma-decay', type=float, default=0,
                        help='Sigma decay rate for each epoch.')
    parser.add_argument('--label-type', metavar='LABELTYPE', default='Gaussian',
                        choices=['Gaussian', 'Cauchy'],
                        help='Labelmap dist type: (default=Gaussian)')
    # Miscs
    parser.add_argument('-c', '--checkpoint', default='checkpoint', type=str, metavar='PATH',
                        help='path to save checkpoint (default: checkpoint)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                        help='evaluate model on validation set')
    parser.add_argument('-d', '--debug', dest='debug', action='store_true',
                        help='show intermediate results')
    parser.add_argument('--display', default=25, type=int, metavar='N',
                        help='how many epochs to display')

    return parser.parse_args()

def main(args):
    global best_nme
    global T_pixel
    T_pixel = args.Temperature_pixel
    global T_pair
    T_pair = args.Temperature_pair

    # create checkpoint dir
    if not isdir(args.checkpoint):
        mkdir_p(args.checkpoint)

    # simple baseline
    config.MODEL.NUM_JOINTS = args.num_classes

    config.MODEL.EXTRA.NUM_LAYERS = 50
    config.MODEL.PRETRAINED_teacher = './pretrained_models/teacher/r50_44deconv_256_365.pth.tar'  # pretrained teacher
    model_teacher = models.res50_44deconv_256_features.get_pose_net(config, True)  # load pretrained teacher model True

    for param in model_teacher.parameters():
        param.requires_grad = False

    # load pretrained imagenet mobilenet prameter True
    config.MODEL.PRETRAINED_MOBILENETV2_STUDENT = './pretrained_models/mobilenet_v2_imagenet.pth.tar'  # imagenet initial
    config.MODEL.EXTRA.NUM_DECONV_FILTERS = [128, 128, 128]
    config.MODEL.EXTRA.NUM_DECONV_KERNELS = [2, 2, 2]  # [4, 4, 4]
    model = models.mobilev2_22deconv_128_pixel_pair_features.get_face_mobilev2_net(config, True) # True means use pretrained student
    print(model)

    # use several GPU:
    if torch.cuda.device_count() > 1:
        print('use {} GPUS.'.format(torch.cuda.device_count()))

    model_teacher = torch.nn.DataParallel(model_teacher).cuda()
    model = torch.nn.DataParallel(model).cuda()

    # define loss function (criterion) and optimizer
    criterion_teather_pixel = torch.nn.MSELoss(size_average=True).cuda()
    criterion_teather_pair = criterion_pair.CriterionSDcos_sig().cuda() #CriterionSDcos_sig

    criterion = torch.nn.MSELoss(size_average=True).cuda()

    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=args.lr,
                                 betas=(0.9, 0.999))

    # optionally resume from a checkpoint
    title = 'cofw'
    if args.resume:
        if isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            args.lr = checkpoint['lr']
            best_nme = checkpoint['best_nme']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
            logger = Logger(join(args.checkpoint, 'log.txt'), title=title, resume=True)
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    else:
        logger = Logger(join(args.checkpoint, 'log.txt'), title=title)
        logger.set_names(
            ['Epoch', 'LR', 'Tr Loss', 'Tr L_hard', 'T L _soft', 'Val Loss', 'Val L_hard',
             'Val L_soft', 'Tr Nme', 'Val Nme'])
    cudnn.benchmark = True
    print('    Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))

    # Data loading code
    train_loader = torch.utils.data.DataLoader(
        datasets.Cofw_no_vector('./datasets/COFW/COFW_train_color.mat',
                                sigma=args.sigma, label_type=args.label_type),
        batch_size=args.train_batch, shuffle=True,
        num_workers=args.workers, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        datasets.Cofw_no_vector('./datasets/COFW/COFW_test_color.mat',
                                sigma=args.sigma, label_type=args.label_type, train=False),
        batch_size=args.test_batch, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    if args.evaluate:
        print('\nEvaluation only')
        val_dataset = datasets.Cofw_no_vector('./datasets/COFW/COFW_test_color.mat',
                                              sigma=args.sigma, label_type=args.label_type, train=False)

        loss, predictions, Nme, failure_rate = evaluate(val_dataset, val_loader, model, model_teacher,criterion,
                                                        criterion_teather_pixel, criterion_teather_pair,
                                                             args.num_classes, args.debug, args.flip)

        print('Test Mean error = ', Nme)
        print('Test Failure rate = ', failure_rate)
        save_pred(predictions, checkpoint=args.checkpoint)
        return

    lr = args.lr
    for epoch in range(args.start_epoch, args.epochs):

        lr = adjust_learning_rate(optimizer, epoch+1, lr, args.schedule, args.gamma)

        # decay sigma
        if args.sigma_decay > 0:
            train_loader.dataset.sigma *= args.sigma_decay
            val_loader.dataset.sigma *= args.sigma_decay

        # train for one epoch
        train_loss, train_loss_hard, train_loss_soft, train_nme = train(train_loader, model, model_teacher, criterion,
                                                                        criterion_teather_pixel, criterion_teather_pair, optimizer,epoch,
                                                args.debug, args.flip)

        # evaluate on validation set
        valid_loss, valid_loss_hard, valid_loss_soft, predictions, valid_nme = validate(val_loader, model, model_teacher, criterion,
                                                                                        criterion_teather_pixel,
                                                                                        criterion_teather_pair, args.num_classes,epoch,
                                                      args.debug, args.flip)

        print(
            '\nEpoch: %d | LR: %.8f | Tr_Loss: %.8f |Tr_L_hard: %.8f |Tr_L_soft: %.8f | Val_Loss: %.8f |Val_L_hard: %.8f |Val_L_soft: %.8f | Tr_nme: %.4f | Val_nme: %.4f'
            % (epoch + 1, lr, train_loss, train_loss_hard, train_loss_soft, valid_loss,
               valid_loss_hard, valid_loss_soft, train_nme, valid_nme))

        train_acc = 0
        valid_acc = 0
        # append logger file
        logger.append(
            [epoch + 1, lr, train_loss, train_loss_hard, train_loss_soft, valid_loss, valid_loss_hard, valid_loss_soft,
             train_nme, valid_nme])

        # remember best acc and save checkpoint
        is_best = valid_nme < best_nme
        best_nme = min(valid_nme, best_nme)

        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_nme': best_nme,
            'optimizer': optimizer.state_dict(),
            'lr': optimizer.param_groups[0]['lr']  # zhaoyang
        }, predictions, is_best, checkpoint=args.checkpoint)

    logger.close()
    # logger.plot(['Train Acc', 'Val Acc'])
    # savefig(os.path.join(args.checkpoint, 'log.eps'))
    print('valid best_nme = ', best_nme)
    print('Finish Training!')


def train(train_loader, model,model_teacher, criterion,criterion_teather_pixel,criterion_teather_pair, optimizer,epoch, debug=False, flip=True):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    losses_hard = AverageMeter()
    losses_soft = AverageMeter()

    # switch to train mode
    model.train()
    model_teacher.eval()

    nme_count = 0
    nme_batch_sum = 0

    end = time.time()

    gt_win, pred_win = None, None
    # bar = Bar('Processing', max=len(train_loader))
    for i, (inputs, target, meta) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        input_var = inputs.cuda()
        target_var = target.cuda()

        # compute output
        output, pixel_features, pair_features = model(input_var)
        loss_hard = criterion(output, target_var)

        loss = loss_hard

        score_map = output.data.cpu()
        # score_map_teacher = output_soft.data.cpu()

        # acc rewrite
        preds = final_preds(score_map, meta['center'], meta['scale'], [64, 64])

        # NME
        nme_batch_sum = nme_batch_sum + cal_nme(preds, meta['pts'])
        nme_count = nme_count + preds.size(0)

        # if debug and epoch%100==0: # visualize groundtruth and predictions  and epoch % args.display == 0
        #     gt_batch_img = batch_with_heatmap(inputs, target)
        #     pred_batch_img = batch_with_heatmap(inputs, score_map_teacher) # score_map_teacher
        #     if not gt_win or not pred_win:
        #         ax1 = plt.subplot(121)
        #         ax1.title.set_text('Train-Groundtruth')
        #         gt_win = plt.imshow(gt_batch_img)
        #         ax2 = plt.subplot(122)
        #         ax2.title.set_text('Prediction')
        #         pred_win = plt.imshow(pred_batch_img)
        #     else:
        #         gt_win.set_data(gt_batch_img)
        #         pred_win.set_data(pred_batch_img)
        #     plt.pause(.05)
        #     plt.draw()

        # measure accuracy and record loss
        losses.update(loss.item(), inputs.size(0))

        losses_hard.update(loss_hard.item(), inputs.size(0))
        # losses_soft.update(loss_soft.item(), inputs.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        # bar.suffix  = '({batch}/{size}) Data: {data:.6f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | Acc: {acc: .4f}'.format(
        #             batch=i + 1,
        #             size=len(train_loader),
        #             data=data_time.val,
        #             bt=batch_time.val,
        #             total=bar.elapsed_td,
        #             eta=bar.eta_td,
        #             loss=losses.avg,
        #             acc=0
        #             )
        # bar.next()

    # bar.finish()
    return losses.avg, losses_hard.avg, losses_soft.avg, nme_batch_sum / nme_count

def validate(val_loader, model,model_teacher, criterion, criterion_teather_pixel,criterion_teather_pair, num_classes,epoch, debug=False, flip=True):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    losses_hard = AverageMeter()
    losses_soft = AverageMeter()

    # predictions
    if num_classes == 29:
        predictions = torch.Tensor(val_loader.dataset.__len__(), num_classes, 2)
    else:
        predictions = torch.Tensor(val_loader.dataset.__len__(), num_classes/2, 2)

    # switch to evaluate mode
    model.eval()
    model_teacher.eval()

    nme_count = 0
    nme_batch_sum = 0

    gt_win, pred_win = None, None
    end = time.time()
    # bar = Bar('Processing', max=len(val_loader))
    for i, (inputs, target, meta) in enumerate(val_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        input_var = inputs.cuda()
        target_var = target.cuda()

        # compute output
        output, pixel_features, pair_features = model(input_var)
        loss_hard = criterion(output, target_var)

        loss = loss_hard

        score_map = output.data.cpu()

        if flip:
            flip_input_var = torch.autograd.Variable(
                    torch.from_numpy(fliplr(inputs.clone().numpy())).float().cuda(), 
                    volatile=True
                )
            flip_output_var = model(flip_input_var)
            flip_output = flip_back(flip_output_var[-1].data.cpu())
            score_map += flip_output

        # generate predictions
        preds = final_preds(score_map, meta['center'], meta['scale'], [64, 64])

        # NME
        nme_batch_sum = nme_batch_sum + cal_nme(preds, meta['pts'])
        nme_count = nme_count + preds.size(0)

        for n in range(score_map.size(0)):
            predictions[meta['index'][n], :, :] = preds[n, :, :]

        if debug and epoch%100==0: # and epoch % args.display == 0
            gt_batch_img = batch_with_heatmap(inputs, target)
            pred_batch_img = batch_with_heatmap(inputs, score_map)
            if not gt_win or not pred_win:
                plt.subplot(121)
                plt.title('Val-Groundtruth')
                gt_win = plt.imshow(gt_batch_img)
                plt.subplot(122)
                plt.title('Prediction')
                pred_win = plt.imshow(pred_batch_img)
            else:
                gt_win.set_data(gt_batch_img)
                pred_win.set_data(pred_batch_img)
            plt.pause(.05)
            plt.draw()

        # measure accuracy and record loss
        losses.update(loss.item(), inputs.size(0))

        losses_hard.update(loss_hard.item(), inputs.size(0))
        # losses_soft.update(loss_soft.item(), inputs.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    #     # plot progress
    #     bar.suffix  = '({batch}/{size}) Data: {data:.6f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | Acc: {acc: .4f}'.format(
    #                 batch=i + 1,
    #                 size=len(val_loader),
    #                 data=data_time.val,
    #                 bt=batch_time.avg,
    #                 total=bar.elapsed_td,
    #                 eta=bar.eta_td,
    #                 loss=losses.avg,
    #                 acc=0
    #                 )
    #     bar.next()
    #
    # bar.finish()

    return losses.avg, losses_hard.avg,losses_soft.avg, predictions, nme_batch_sum / nme_count

def evaluate(val_dataset,val_loader, model,model_teacher, criterion,criterion_teather_pixel,criterion_teather_pair, num_classes, debug=False, flip=True):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    losses_hard = AverageMeter()
    losses_soft = AverageMeter()

    # predictions
    if num_classes == 29:
        predictions = torch.Tensor(val_loader.dataset.__len__(), num_classes, 2)
    else:
        predictions = torch.Tensor(val_loader.dataset.__len__(), num_classes/2, 2)

    # switch to evaluate mode
    model.eval()
    model_teacher.eval()

    nme_count = 0
    nme_batch_sum = 0

    count_failure = 0

    gt_win, pred_win = None, None
    end = time.time()
    # bar = Bar('Processing', max=len(val_loader))

    times = 0
    iii = 0

    for i, (inputs, target, meta) in enumerate(val_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        input_var = inputs.cuda()
        target_var = target.cuda()

        # compute output
        test_time = time.time()
        output, pixel_features, pair_features = model(input_var)
        torch.cuda.synchronize()
        tttime = time.time() - test_time

        times = times + tttime
        iii = iii + 1

        loss_hard = criterion(output, target_var)

        loss = loss_hard

        score_map = output.data.cpu()
        # score_map_teacher = output_soft.data.cpu()

        if flip:
            flip_input_var = torch.autograd.Variable(
                    torch.from_numpy(fliplr(inputs.clone().numpy())).float().cuda(),
                    volatile=True
                )
            flip_output_var = model(flip_input_var)
            flip_output = flip_back(flip_output_var[-1].data.cpu())
            score_map += flip_output

        # generate predictions
        preds = final_preds(score_map, meta['center'], meta['scale'], [64, 64])

        # NME
        nme_temp = cal_nme(preds, meta['pts'])
        print(nme_temp)

        if nme_temp > 0.1:
            count_failure += 1

        nme_batch_sum = nme_batch_sum + nme_temp
        nme_count = nme_count + preds.size(0)

        for n in range(score_map.size(0)):
            predictions[meta['index'][n], :, :] = preds[n, :, :]

        # # show
        # img_show = val_dataset.get_img(i)
        # plt.figure()
        # plt.imshow(img_show)
        # plt.scatter(np.array(preds.numpy())[0][:,0], np.array(preds.numpy())[0][:,1], s=10, marker='.', c='r')
        # plt.scatter(np.array(meta['pts'])[0][:, 0], np.array(meta['pts'])[0][:, 1], s=10, marker='.', c='g')
        # plt.title('r is predict, nme = ' + str(nme_temp))
        # plt.savefig('./save_img_challenge/' + str(i) + '.jpg')
        # # plt.show()
        # plt.close()

        if debug: # and epoch % args.display == 0
            gt_batch_img = batch_with_heatmap(inputs, target)
            pred_batch_img = batch_with_heatmap(inputs, score_map) # score_map
            if not gt_win or not pred_win:
                plt.subplot(121)
                plt.title('Val-Groundtruth')
                gt_win = plt.imshow(gt_batch_img)
                plt.subplot(122)
                plt.title('Prediction')
                pred_win = plt.imshow(pred_batch_img)
            else:
                gt_win.set_data(gt_batch_img)
                pred_win.set_data(pred_batch_img)
            plt.pause(.05)
            plt.draw()

        # measure accuracy and record loss
        losses.update(loss.item(), inputs.size(0))

        losses_hard.update(loss_hard.item(), inputs.size(0))
        # losses_soft.update(loss_soft.item(), inputs.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    #     # plot progress
    #     bar.suffix  = '({batch}/{size}) Data: {data:.6f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | Acc: {acc: .4f}'.format(
    #                 batch=i + 1,
    #                 size=len(val_loader),
    #                 data=data_time.val,
    #                 bt=batch_time.avg,
    #                 total=bar.elapsed_td,
    #                 eta=bar.eta_td,
    #                 loss=losses.avg,
    #                 acc=0
    #                 )
    #     bar.next()
    #
    # bar.finish()
    print('count_failure = ', count_failure)
    print('test time = ', times/iii)

    return losses.avg, predictions, nme_batch_sum / nme_count, count_failure / nme_count

if __name__ == '__main__':

    args = parse()
    main(args)