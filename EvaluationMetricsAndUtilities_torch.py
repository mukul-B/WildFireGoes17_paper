# -*- coding: utf-8 -*-
"""
This script will perform Evalution operation like Get IOu between two images (tensor), get PSNR , get Otsu thresholding
Created on Sun Jul 26 11:17:09 2020

@author:  mukul badhan
on Sun Jul 23 11:17:09 2022
"""

import math
import numpy as np
import torch
import math
import numpy as np
import cv2


# get PSNR between two images
def psnr(target, output, max_val=1.0):
    mse = torch.mean((target - output) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(max_val / math.sqrt(mse))

# get IOU between two images
def get_IOU(target, pred):
    pred_binary = pred.clone()
    target_binary = target.clone()
    pred_binary[pred_binary != 0] = 1
    target_binary[target_binary != 0] = 1
    intersection = (pred_binary * target_binary).sum()
    total = (pred_binary + target_binary).sum()
    union = total - intersection
    IOU = (intersection) / (union)
    # IOU = (intersection + SMOOTH) / (union + SMOOTH)
    return IOU.item()

# get PSNR between two images but only considering Union of two image
def PSNR_union(target, pred,max_val=367.0):
    imdff = pred - target
    union = pred + target
    imdff[union == 0] = 0
    imdff = imdff.flatten()
    pixelsInIntersection = torch.count_nonzero(union).item()

    if pixelsInIntersection > 0:
        rmse = torch.sqrt(torch.sum(imdff ** 2) / pixelsInIntersection).item()
    else:
        return 0
    if rmse == 0:
        return 100
    return 20 * torch.log10(torch.tensor(max_val) / rmse).item()

# get PSNR between two images but only considering Intersection of two image
def PSNR_intersection(target, pred,max_val=367.0):
    imdff = pred - target
    interaction = pred * target
    imdff[interaction == 0] = 0
    imdff = imdff.flatten()
    pixelsInIntersection = torch.count_nonzero(interaction).item()
    if pixelsInIntersection > 0:
        rmse = torch.sqrt(torch.sum(imdff ** 2) / pixelsInIntersection).item()
    else:
        return 0
    if rmse == 0:
        return 100
    return 20 * torch.log10(torch.tensor(max_val) / rmse).item()

# get coverage: area of fire per area of window
def get_coverage(img):
    return (torch.count_nonzero(img) / img.numel()).item()

def show_img(axis, img, label):
    # axis.imshow(img)
    axis.imshow(img, cmap='gray')
    axis.set_title(label, size=8)


def show_hist(axis, title, binsl, hist, minr_tick):
    axis.hist(binsl[:-1], bins=binsl, weights=hist, color='blue')
    # # axis.set_title(title, size=8)
    # axis.set_xticks(minr_tick, minor=True, color='red')
    # axis.set_xticklabels(minr_tick, fontdict=None, minor=True, color='red', size=13)
    axis.tick_params(axis='x', which='minor', colors='red', size=13)
    axis.set_ylabel("Pixels Fraction", fontsize=15)
    axis.set_xlabel('Normalized Radiance', fontsize=15)
    axis.set_xlim(0, )


# get the othsu thresholding of an image
def getth(image, on=0):
    bins = 413
    device = image.device
    image_r = image.clone()
    image_r = image_r * (bins-1)
    # Get the image histogram
    
    max_val , min_val = image_r.max().item(), image_r.min().item()
    bin_width = (max_val - min_val) / bins
    hist = torch.histc(image_r, bins=bins).to(device)
    # bin_edges = torch.arange(min_val, max_val , bin_width)
    
    # hist = torch.histc(image_r, bins=bins, min=0, max=bins-1).to(device)
    bin_edges = torch.linspace(min_val, max_val, bins + 1).to(device)
    if on:
        hist, bin_edges = hist[on:], bin_edges[on:]
    
    # Calculate bin midpoints
    bin_mids = (bin_edges[:-1] + bin_edges[1:]) / 2.
    weight1 = torch.cumsum(hist, dim=0)
    weight2 = torch.cumsum(hist.flip(0), dim=0).flip(0)
    # Calculate class means (mean1 and mean2)
    mean1 = torch.cumsum(hist * bin_mids, dim=0) / weight1
    mean2 = (torch.cumsum((hist * bin_mids).flip(0), dim=0) / weight2.flip(0)).flip(0)
    # Inter-class variance (Otsu's method for thresholding)
    inter_class_variance = weight1[:-1] * weight2[1:] * (mean1[:-1] - mean2[1:]) ** 2
    # Find index of maximum inter_class_variance
    index_of_max_val = torch.argmax(inter_class_variance)
    threshold = bin_mids[:-1][index_of_max_val]
    threshold = threshold + on if ((threshold + on) < (bins-1)) else threshold
    image_r[image_r < threshold] = 0
    image_r[image_r >= threshold] = 1
    return round(threshold.item(), 2), image_r, hist, bin_edges, index_of_max_val.item()

# get the besr othsu thresholding of an image: recursively applying thresholding to get the max IOU
def best_threshold_iteration(groundTruth, input):
    
    maxiou = 0
    level = 0
    pth2 = torch.ones_like(input)
    pret2 = 0
    phist2, pbin_edges = None, None
    imgs = []
    thr_list = []
    iou_list = []
    l = 0
    pcoverage = 0
    while (True):
        ret2, th2, hist2, bin_edges2, index_of_max_val2 = getth(pth2 * input, on=int(pret2))
        iou_i = get_IOU(groundTruth, th2)

        if maxiou >= iou_i:
            break
        maxiou = iou_i
        level += 1
        pret2, pth2, phist2, pbin_edges = ret2, th2, hist2, bin_edges2
        # plt.plot(pth2)
        # plt.savefig(f"checko{level}.png")
        # input()

    ret2, th2, hist2, bin_edges2 = pret2, pth2, phist2, pbin_edges
    return level, ret2, th2


def noralize_goes_to_radiance(ngf, gf_max, gf_min):
    color_normal_value = 1
    return (gf_min + (ngf * ((gf_max - gf_min) / color_normal_value))).round(5)


def noralize_viirs_to_radiance(nvf, vf_max,vf_min=0):
    color_normal_value = 1
    return (vf_min + (nvf * ((vf_max-vf_min) / color_normal_value))).round(2)

def denoralize(value, v_max, v_min):
    color_normal_value = 1
    return (v_min + (value * ((v_max - v_min) / color_normal_value))).round(2)