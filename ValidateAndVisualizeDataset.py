"""
This script will have matrix to evalues GOES and VIIRS final dataset
checking dimention of image
checking signal to noise ratio
visualising them

Created on Sun Jul 23 11:17:09 2022

@author: mukul
"""
import math
import os

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from PIL import Image

# from CreateDatasetPipeline import prepareDirectory
from GlobalValues import GOES_MAX_VAL, VIIRS_UNITS, GOES_product_size, viirs_dir, goes_dir, compare_dir
import datetime

from WriteDataset4DLModel import Normalize_img
# from CommonFunctions import prepareDirectory
from SiteInfo import SiteInfo

# demonstrate reference_data standardization with sklearn


def prepareDirectory(path):
    os.makedirs(path, exist_ok=True)

def getth(image, on=0):
    # bins= 413
    # print(max([ image[i].max() for i in range(len(image))]) != image.max())
    
    bins = int(image.max()-image.min() + 1)
    # on = int(image.min())
    # Set total number of bins in the histogram
    image_r = image.copy()
    # image_r = image_r * (bins-1)
    # Get the image histogram
    hist, bin_edges = np.histogram(image_r, bins=bins)
    if (on):
        hist, bin_edges = hist[on:], bin_edges[on:]
    # Get normalized histogram if it is required

    # Calculate centers of bins
    bin_mids = (bin_edges[:-1] + bin_edges[1:]) / 2.
    # Iterate over all thresholds (indices) and get the probabilities w1(t), w2(t)
    weight1 = np.cumsum(hist)
    weight2 = np.cumsum(hist[::-1])[::-1]
    # Get the class means mu0(t)
    mean1 = np.cumsum(hist * bin_mids) / weight1
    # Get the class means mu1(t)
    mean2 = (np.cumsum((hist * bin_mids)[::-1]) / weight2[::-1])[::-1]
    inter_class_variance = weight1[:-1] * weight2[1:] * (mean1[:-1] - mean2[1:]) ** 2
    # Maximize the inter_class_variance function val
    index_of_max_val = np.argmax(inter_class_variance)
    threshold = bin_mids[:-1][index_of_max_val]
    threshold = threshold + on if ((threshold + on) < (bins-1)) else threshold
    image_r[image_r < threshold] = 0
    image_r[image_r >= threshold] = 1
    return round(threshold, 2), image_r, hist, bin_edges, index_of_max_val


# visualising GOES and VIIRS
def viewtiff(location,v_file, g_file, date, save=True, compare_dir=None):
    VIIRS_data = xr.open_rasterio(v_file)
    GOES_data = xr.open_rasterio(g_file)

    vd = VIIRS_data.variable.data[0]
    FRP = VIIRS_data.variable.data[1]
    gd = [GOES_data.variable.data[i] for i in range(GOES_product_size)]
    # if in GOES and VIIRS , the values are normalized, using this flag to visualize result
    normalized = False
    vmin,vmax = (0, 250) if normalized else (200,420)
    to_plot, lables = basic_plot(vd, gd)
    # to_plot, lables = multi_spectralAndFRP(vd, FRP, gd)

    site = SiteInfo(location)
    longitude = site.longitude
    latitude = site.latitude
    compare_dir = compare_dir.replace( location,'')

    # small_big = "small" if np.count_nonzero(vd) < 30 else "big"
    # small_big = 'issueGOES'
    # compare_dir = compare_dir.replace('compare','compare_'+small_big)
    # compare_dir = f'{compare_dir}/{str(site.EPSG)}'
    # compare_dir = f'{compare_dir}/c{int((latitude // 4 ) * 4)}_{int(longitude)}'
    # compare_dir = f'{compare_dir}/{int(longitude)}_{int(latitude)}_{location}'
    compare_dir = f'{compare_dir}/{location}'
    # compare_dir = f'{compare_dir}/{int(20 *(FRP.sum() // 20))}'
    
    save_path = f'{compare_dir}/{int(longitude)}_{int(latitude)}_{location}_{date}.png' if save else None
    # save_path = f'{compare_dir}/{str(site.EPSG)}/{int(longitude)}_{int(latitude)}_{location}_{date}.png' if save else None
    plot_condition = True
    # plot_condition = (np.count_nonzero(gd[0]==0) > 5)
    # plot_condition = (np.count_nonzero(vd) < 30 and FRP.sum() <150 )
    if(plot_condition):
        prepareDirectory(compare_dir)
        plot_title = f'{location} at {date} coordinates : {longitude},{latitude}'
        print(save_path)
        Plot_list(plot_title,  to_plot, lables, vd.shape, None, None, save_path)

def basic_plot(vd, gd):
    to_plot = [gd[0],vd,(gd[0] - vd)]
    lables = ["GOES","VIIRS","VIIRS On GOES"]
    return to_plot,lables

def multi_spectralAndFRP(vd, FRP, gd):
    Active_fire = (gd[0]-gd[1])/(gd[0]+gd[1])
    cloud_remove_280 = Active_fire * (gd[2]> 280) * 1000
    to_plot = [gd[0],vd,(gd[0] - vd),FRP,Active_fire,cloud_remove_280]
    lables = ["GOES","VIIRS "+str(vd.sum())+' '+str(np.count_nonzero(vd))+' '+str(round(np.average(vd),2)),"VIIRS On GOES",'FRP '+str(FRP.sum())+' '+str(np.count_nonzero(FRP))+' '+str(round(np.max(FRP),2)),"Active_fire",'cloud_remove_280']
    return to_plot,lables

def multi_spectral_plots(vd, gd):
    Active_fire = (gd[0]-gd[1])/(gd[0]+gd[1])
    cloud_remove_280 = Active_fire * (gd[2]> 280) * 1000
    # ret1, th1, hist1, bins1, index_of_max_val1 = getth(cloud_remove_280, on=0)
    # print(cloud_remove_280.min(),cloud_remove_280.max(),ret1)
    # cloud_remove = Active_fire * (cloud_remove_280 > ret1) * 1000
    cloud_remove = cloud_remove_280 * (cloud_remove_280 > 0) 
    cloud_remove = (cloud_remove * GOES_MAX_VAL)  / cloud_remove.max()
    gd[0] = Normalize_img(gd[0])
    cloud_remove = Normalize_img(cloud_remove,gf_min = 0, gf_max = GOES_MAX_VAL)
    to_plot = [gd[0],Active_fire,cloud_remove_280,cloud_remove,vd]
    lables = ["GOES","Active_fire","cloud_remove_280","cloud_remove","VIIRS"]
    
    # to_plot = [gd[0],Active_fire,cloud_remove_280,vd]
    # lables = ["GOES","Active_fire","Active_fire with Cloud Mask","VIIRS"]
    return to_plot,lables

def Plot_list( title, to_plot, lables, shape, vmin=None, vmax=None, save_path=None):

    X, Y = np.mgrid[0:1:complex(str(shape[0]) + "j"), 0:1:complex(str(shape[1]) + "j")]
    n = len(to_plot)
    fig, ax = plt.subplots(1, n, constrained_layout=True, figsize=(4*n, 4))
    fig.suptitle(title)

    for k in range(n):
        curr_img = ax[k] if n > 1 else ax
        p = curr_img.pcolormesh(Y, -X, to_plot[k], cmap="jet", vmin=vmin, vmax=vmax)
        curr_img.tick_params(left=False, right=False, labelleft=False,
                          labelbottom=False, bottom=False)
        curr_img.text(0.5, -0.1, lables[k], transform=curr_img.transAxes, ha='center', fontsize=12)

        cb = fig.colorbar(p, pad=0.01)
        cb.ax.tick_params(labelsize=11)
        cb.set_label(VIIRS_UNITS, fontsize=12)
    plt.rcParams['savefig.dpi'] = 600
    if (save_path):
        fig.savefig(save_path)
        plt.close()
    plt.show()


def plot_individual_images(X, Y, compare_dir, g_file, gd, vd):
    ar = [vd, gd, gd - vd]
    if (g_file == 'reference_data/Kincade/GOES/ABI-L1b-RadC/tif/GOES-2019-10-27_949.tif'):
        print(g_file, compare_dir)
        for k in range(3):
            fig2 = plt.figure()
            ax = fig2.add_subplot()
            a = ax.pcolormesh(Y, -X, ar[k], cmap="jet", vmin=200, vmax=420)
            cb = fig2.colorbar(a, pad=0.01)
            cb.ax.tick_params(labelsize=11)
            cb.set_label('Radiance (K)', fontsize=12)
            plt.tick_params(left=False, right=False, labelleft=False,
                            labelbottom=False, bottom=False)
            # plt.show()
            plt.savefig(f'{compare_dir}/data_preprocessing{k}.png', bbox_inches='tight', dpi=600)
            plt.close()



def PSNR(pred, gt, shave_border=0):
    imdff = pred - gt
    print(imdff)

    imdff = imdff.flatten()
    rmse = math.sqrt(np.mean(np.array(imdff ** 2)))
    if rmse == 0:
        return 100
    return 20 * math.log10(255.0 / rmse)


def shape_check(v_file, g_file):
    VIIRS_data = xr.open_rasterio(v_file)
    GOES_data = xr.open_rasterio(g_file)

    vf = VIIRS_data.variable.data[0]
    gd = GOES_data.variable.data[0]
    vf = np.array(vf)[:, :]
    gd = np.array(gd)[:, :]
    # (343,)(27, 47)
    if(vf.shape != gd.shape):
        print(vf.shape, gd.shape)
    # print(PSNR(gd, vf))


# the dataset created is evaluated visually and statistically
def validateAndVisualizeDataset(location, product):
    # product_name = product['product_name']
    # band = product['band']
    viirs_tif_dir = viirs_dir.replace('$LOC', location)
    product_band = ''.join(map(lambda item: f"{item['product_name']}{format(item['band'],'02d')}", product))
    goes_tif_dir = goes_dir.replace('$LOC', location).replace('$PROD_BAND', product_band)
    # goes_tif_dir = goes_dir.replace('$LOC', location).replace('$PROD', product['product_name']).replace('$BAND', format(product['band'],'02d'))
    comp_dir = compare_dir.replace('$LOC', location)
    viirs_list = os.listdir(viirs_tif_dir)
    for v_file in viirs_list:
        g_file = "GOES" + v_file[10:]
        sample_date = v_file[11:-4]
        sample_date = sample_date.split('_')
        sample_date = f'{sample_date[0]}_{sample_date[1].rjust(4,"0")}'
        # shape_check(viirs_tif_dir + v_file, goes_tif_dir + g_file)
        viewtiff(location,viirs_tif_dir + v_file, goes_tif_dir + g_file, sample_date, compare_dir=comp_dir, save=True)