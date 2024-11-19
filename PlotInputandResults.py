"""
This script provide encapsulating class and function to scalably plot multiple images in given orientation

Created on Sun june 23 11:17:09 2023

@author: mukul
"""

import matplotlib.pyplot as plt
import numpy as np

from EvaluationMetricsAndUtilities_numpy import denoralize, getth

# plt.style.use('plot_style/wrf')


class ImagePlot:
    def __init__(self,unit,vmax,vmin,image_blocks,lable_blocks,binary=False):
        self.unit = unit
        self.vmin = vmin
        self.vmax = vmax
        self.binary = binary
        if(self.unit):
            self.image_blocks = denoralize(image_blocks,vmax,vmin)
        else:
            self.image_blocks = None
        self.lable_blocks = lable_blocks

def plot_from_ImagePlot(title,img_seq,path,shape=(128,128),colection=True):
    
    X, Y = np.mgrid[0:1:complex(str(shape[0]) + "j"), 0:1:complex(str(shape[1]) + "j")]
    c,r = len(img_seq) ,len(img_seq[0])
    if(colection):
        fig, axs = plt.subplots(c, r, constrained_layout=True, figsize=(12, 4*c))
        fig.suptitle(title)
    for col in range(c):
        for row in range(r):
            curr_image = img_seq[col][row]
            image_blocks = curr_image.image_blocks
            lable_blocks = curr_image.lable_blocks
            binary = curr_image.binary
            cb_unit = curr_image.unit
            vmin,vmax = curr_image.vmin, curr_image.vmax
            # vmin,vmax = 0 , 413
            if(colection):
                if(r == 1):
                    ax = axs[col]
                elif(c==1):
                    ax = axs[row]
                else:
                    ax = axs[col][row]
                ax.set_title(lable_blocks)
                

            if  image_blocks is not None:
                if(not colection):
                    fig = plt.figure()
                    # fig.suptitle(title)
                    ax = fig.add_subplot()

                if binary:
                    sc = ax.pcolormesh(Y, -X, image_blocks, cmap=plt.get_cmap("gray_r", 2), vmin=vmin, vmax=vmax)
                else:
                    sc = ax.pcolormesh(Y, -X, image_blocks, cmap=plt.get_cmap("jet"), vmin=vmin, vmax=vmax)
                    cb = fig.colorbar(sc, pad=0.01,ax=ax)
                    cb.ax.tick_params(labelsize=11)
                    cb.set_label(cb_unit, fontsize=13)
                
                # plt.tick_params(left=False, right=False, labelleft=False,
                #                 labelbottom=False, bottom=False)

                # Hide X and Y axes tick marks
                ax.set_xticks([])
                ax.set_yticks([])
                
                if(not colection):
                    # pl = path.split('/')
                    # filename = pl[-1].replace('.png','')
                    filenamecorrection = lable_blocks.replace('\n','_').replace(' ','_').replace('.','_').replace('(','_').replace(')','_').replace(':','_')
                    # path ='/'.join(pl[:-1]) + f'/{filename}_{filenamecorrection}.png'
                    path = path.replace('.png',f'_{filenamecorrection}.png')
                    # print(path)
                    # prepareDirectory(path)
                    plt.savefig(path,
                                bbox_inches='tight', dpi=600)
                    plt.show()
                    plt.close()
    if(colection):
        plt.rcParams['savefig.dpi'] = 600
        fig.savefig(path)
        plt.close()
        
def plot_histogramandimage(image,path):
    # image = np.random.randint(0, 256, size=(256, 256), dtype=np.uint8)
    # histogram, bins = np.histogram(image, bins=256, range=(0, 256))
    ret2, th2, hist2, bins2, index_of_max_val2 = getth(image)

    

    # Create subplots with constrained layout
    fig, axs = plt.subplots(1, 2, constrained_layout=True, figsize=(12, 8))

    # Plot the image
    axs[0].imshow(image, cmap='gray')
    axs[0].set_title("Image")
    axs[0].axis('off')

    # Plot the histogram
    # axs[1].plot(hist2, color='black')
    axs[1].hist(bins2[1:-1], bins=bins2[1:], weights=hist2[1:], color='blue')
    axs[1].set_title("Histogram")
    axs[1].set_xlabel("Pixel Value")
    axs[1].set_ylabel("Frequency")
    axs[1].set_xlim(0, 255)
    # axs[1].set_ylim(0, 200)

    # show_hist([1], "Histogram", bins2, hist2, 1)
    

    plt.tight_layout()
    plt.savefig(path)
    plt.close()