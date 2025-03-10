"""
This script will run through the directory of training images, load
the image pairs

Created on Sun Jul 23 11:17:09 2022

@author: mukul
"""

import os
import pandas as pd
import time
import multiprocessing as mp
from tqdm import tqdm

from CreateTrainingDataset import createDataset
from ValidateAndVisualizeDataset import validateAndVisualizeDataset
from WriteDataset4DLModel import writeDataset
from GlobalValues import  GOES_product, toExecuteSiteList, write_training_dir,testing_dir


from GlobalValues import compare_dir,goes_dir,viirs_dir,logs, testing_dir, write_training_dir, GOES_ndf,seperate_th,goes_OrtRec_dir

#prepare directories for writting Files
def prepareDir(location, product):
    product_band = ''.join(map(lambda item: f"{item['product_name']}{format(item['band'],'02d')}", product))
    goes_tif_dir = goes_dir.replace('$LOC', location).replace('$PROD_BAND', product_band)
    goes_OrtRec_tif_dir = goes_OrtRec_dir.replace('$LOC', location).replace('$PROD_BAND', product_band)
    viirs_tif_dir = viirs_dir.replace('$LOC', location)
    comp_dir = compare_dir.replace('$LOC', location)

    os.makedirs(goes_tif_dir, exist_ok=True)
    os.makedirs(goes_OrtRec_tif_dir, exist_ok=True)
    os.makedirs(viirs_tif_dir, exist_ok=True)
    os.makedirs(comp_dir, exist_ok=True)
    os.makedirs(logs, exist_ok=True)
    os.makedirs(write_training_dir, exist_ok=True)
    os.makedirs(testing_dir, exist_ok=True)
    os.makedirs(GOES_ndf, exist_ok=True)

def prepareDirectory(path):
    os.makedirs(path, exist_ok=True)


def on_success(location):
    pbar.update(1)
    print(f"{location} processed successfully at {time.time() - start_time:.2f} seconds")

def on_error(e):
    print(f"Error: {e}")

# Create intermediate TIFF files that are consistently resampled and reprojected
# Writen in nupmy data to be read by Deep learning Model
def CreateDatasetPipeline(location, product, train_test):
    prepareDir(location, product)
    # createDataset(location, product)
    validateAndVisualizeDataset(location, product)
    # writeDataset(location, product, train_test)
    return location

# check the count of training data written by pipeline
def count_training_set_created(dir):
    files_and_dirs = os.listdir(dir)
    file_count = sum(os.path.isfile(os.path.join(dir, item)) for item in files_and_dirs)

    print(f'{file_count} records writtenin {dir}')
    

if __name__ == '__main__':
    print(f'Creating dataset for sites in {toExecuteSiteList}')
    data = pd.read_csv(toExecuteSiteList)
    locations = data["Sites"][:]
    train_test = write_training_dir
    start_time = time.time()
    # train_test = testing_dir
    # pipeline run for sites mentioned in toExecuteSiteList
    parallel = 0
    if(parallel):
        pool = mp.Pool(8)
    # Initialize tqdm progress bar
    with tqdm(total=len(locations)) as pbar:
        results = []
        for location in locations:
            if(parallel):
                result = pool.apply_async(CreateDatasetPipeline, args=(location, GOES_product, train_test), 
                                      callback=on_success, error_callback=on_error)
            else:
                result = CreateDatasetPipeline(location, GOES_product, train_test)
                on_success(location)
        if(parallel):
            pool.close()
            pool.join()
    count_training_set_created(train_test)
    
