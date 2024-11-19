"""
This script will
create (per site per date) tiff from VIIRS csv
and download corresponding GOES file and crop it for site bounding box and create TIFF

Created on Sun Jul 23 11:17:09 2022

@author: mukul
"""
import datetime
import os

from GlobalValues import  goes_dir, goes_OrtRec_dir

from GoesProcessing import GoesProcessing
from SiteInfo import SiteInfo
from VIIRSProcessing import VIIRSProcessing

# com_log = open("log_path_G171.txt", 'a')
def count_training_set_created(dir):
    files_and_dirs = os.listdir(dir)
    # Count only the files (not directories)
    file_count = sum(os.path.isfile(os.path.join(dir, item)) for item in files_and_dirs)

    return file_count

# 1) check what all time staps are avaiable for a wildwire event from VIIRS csv
# 2) download GOES images closest to these timestamps
# 3) create tiff files for wildfire evnts from GOES and VIIRS
def createDataset(location, product):
    site = SiteInfo(location)
    start_time, end_time = site.start_time, site.end_time
    time_dif = end_time - start_time
    # TODO: need to add 1 in time_diff because it is ignoring last date
    log_path = 'logs/failures_' + location + '_' + str(site.start_time) + '_' + str(site.end_time) + '.txt'
    product_band = ''.join(map(lambda item: f"{item['product_name']}{format(item['band'],'02d')}", product))
    goes_tif_dir = goes_dir.replace('$LOC', location).replace('$PROD_BAND', product_band)
    goes_OrtRec_tif_dir = goes_OrtRec_dir.replace('$LOC', location).replace('$PROD_BAND', product_band)
    site.get_image_dimention()
    # initialize Goes object and prvide file for log
    goes = GoesProcessing(log_path,list(map(lambda item: item['product_name'], product)),list(map(lambda item: item['band'],product))
                                    ,site=site)
    # initialize VIIRS object , this will create firefixel for particular site and define image parameters
    v2r_viirs = VIIRSProcessing(year=str(start_time.year), satellite="viirs-snpp", site=site)
    v2r_viirs.extract_hotspots()

    # running for each date
    for i in range(time_dif.days):
        fire_date = str(start_time + datetime.timedelta(days=i))

        # filter firepixel for a date
        unique_time = v2r_viirs.get_unique_dateTime(fire_date)
        unique_time = v2r_viirs.collapse_close_dates(unique_time)
        # running for ever hhmm for perticular date
        for ac_time in unique_time:
            paths = goes.download_goes(fire_date, str(ac_time))
            if -1 not in paths:
                v2r_viirs.make_tiff(fire_date, ac_time)
                goes.nc2tiff(fire_date, ac_time, paths, site, site.image_size, goes_tif_dir)
    # print(location,count_training_set_created(goes_tif_dir))