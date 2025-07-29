#-*- coding:utf-8 -*-
import os
import os.path as osp
import sys
import time
import glob
import logging
import argparse
from collections import OrderedDict
from torch.utils.data import DataLoader
from dataset.dataset_sig17 import SIG17_Test_Dataset, SIG17_Validation_Dataset
from models.hdr_hpb import HDR_HPB
from utils.utils import *


parser = argparse.ArgumentParser(description="Test Setting")
parser.add_argument("--dataset_dir", type=str, default='E:\Datasets\Sig17',
                        help='dataset directory')
parser.add_argument('--no_cuda', action='store_true', default=False,
                        help='disables CUDA training')
parser.add_argument('--test_batch_size', type=int, default=1, metavar='N',
                        help='testing batch size (default: 1)')
parser.add_argument('--num_workers', type=int, default=1, metavar='N',
                        help='number of workers to fetch data (default: 1)')
parser.add_argument('--patch_size', type=int, default=256)
parser.add_argument('--pretrained_model', type=str, default='./checkpoints/best_checkpoint.pth')
# parser.add_argument('--pretrained_model', type=str, default='./checkpoints/ahdr_model.pt')
parser.add_argument('--test_best', action='store_true', default=False)
parser.add_argument('--save_results', action='store_true', default=False)
parser.add_argument('--save_dir', type=str, default="./results/hdr_transformer")
parser.add_argument('--model_arch', type=int, default=0)

def main():
    # Settings
    args = parser.parse_args()

    # pretrained_model
    print(">>>>>>>>> Start Testing >>>>>>>>>")
    print("Load weights from: ", args.pretrained_model)

    # cuda and devices
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')

    # model architecture
    model_dict = {
         0: HDR_HPB(6, 64, 'haar'),
    }
    model = model_dict[args.model_arch].to(device)
    # model = nn.DataParallel(model)
    # model.load_state_dict(torch.load(args.pretrained_model)['state_dict'])


    # state_dict = torch.load(args.pretrained_model)['state_dict']
    state_dict = torch.load(args.pretrained_model, map_location=torch.device('cpu'))['state_dict']
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]  # 去掉前面的 "module."
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)

    # # ahdr cheakpoint
    # checkpoint = torch.load('./checkpoints//val_latest_checkpoint.pth', map_location=torch.device('cpu'))
    # checkpoint['state_dict'] = {k.replace('module.', ''): v for k, v in checkpoint['state_dict'].items()}
    # model.load_state_dict(checkpoint['state_dict'])

    model.eval()

    datasets = SIG17_Validation_Dataset(root_dir=args.dataset_dir, is_training=False, crop=False, crop_size=512)
    dataloader = DataLoader(dataset=datasets, batch_size=1, num_workers=1, shuffle=False)
    psnr_l = AverageMeter()
    ssim_l = AverageMeter()
    psnr_mu = AverageMeter()
    ssim_mu = AverageMeter()
    for idx, img_dataset in enumerate(dataloader):
        # pred_img, label = test_single_img(model, img_dataset, device)
        with torch.no_grad():
            batch_ldr0, batch_ldr1, batch_ldr2, label = img_dataset['input0'].to(device), \
                                                img_dataset['input1'].to(device), \
                                                img_dataset['input2'].to(device), \
                                                img_dataset['label'].to(device),
            # pred_img = model(batch_ldr0, batch_ldr1, batch_ldr2)
            pred_img = model(batch_ldr0, batch_ldr1, batch_ldr2, save_flow=True, flow_prefix=str(idx))
            pred_img = torch.squeeze(pred_img.detach().cpu()).numpy().astype(np.float32)
            label = torch.squeeze(label.detach().cpu()).numpy().astype(np.float32)
        pred_hdr = pred_img.copy()
        pred_hdr = pred_hdr.transpose(1, 2, 0)[..., ::-1]
        # psnr-l and psnr-\mu
        # scene_psnr_l = compare_psnr(label, pred_img, data_range=1.0)
        scene_psnr_l = peak_signal_noise_ratio(label, pred_img, data_range=1.0)
        label_mu = range_compressor(label)
        pred_img_mu = range_compressor(pred_img)
        # scene_psnr_mu = compare_psnr(label_mu, pred_img_mu, data_range=1.0)
        scene_psnr_mu = peak_signal_noise_ratio(label_mu, pred_img_mu, data_range=1.0)
        # ssim-l
        pred_img = np.clip(pred_img * 255.0, 0., 255.).transpose(1, 2, 0)
        label = np.clip(label * 255.0, 0., 255.).transpose(1, 2, 0)
        scene_ssim_l = calculate_ssim(pred_img, label)
        # ssim-\mu
        pred_img_mu = np.clip(pred_img_mu * 255.0, 0., 255.).transpose(1, 2, 0)
        label_mu = np.clip(label_mu * 255.0, 0., 255.).transpose(1, 2, 0)
        scene_ssim_mu = calculate_ssim(pred_img_mu, label_mu)

        psnr_l.update(scene_psnr_l)
        ssim_l.update(scene_ssim_l)
        psnr_mu.update(scene_psnr_mu)
        ssim_mu.update(scene_ssim_mu)

        # save results
        if args.save_results:
            if not osp.exists(args.save_dir):
                os.makedirs(args.save_dir)
            save_hdr(os.path.join(args.save_dir, '{}_pred.hdr'.format(idx)), pred_hdr)

    print("Average PSNR_mu: {:.4f}  PSNR_l: {:.4f}".format(psnr_mu.avg, psnr_l.avg))
    print("Average SSIM_mu: {:.4f}  SSIM_l: {:.4f}".format(ssim_mu.avg, ssim_l.avg))
    print(">>>>>>>>> Finish Testing >>>>>>>>>")


if __name__ == '__main__':
    main()




