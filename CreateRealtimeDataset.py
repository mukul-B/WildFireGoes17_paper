"""
This script will
create (per site per date) tiff from VIIRS csv
and download corresponding GOES file and crop it for site bounding box and create TIFF

Created on Sun Jul 23 11:17:09 2022

@author: mukul
"""
import datetime


from GlobalValues import RAD, RealTimeIncoming_files , viirs_folder, goes_folder , realtime_site_conf
from GoesProcessing import GoesProcessing
from RadarProcessing import RadarProcessing
from SiteInfo import SiteInfo
from VIIRSProcessing import VIIRSProcessing
import datetime as dt


def create_dummy_uniquetime(hour_frequency=1,minute_frequency=5):
    return [str(h).zfill(2) + str(m).zfill(2) for h in range(0, 24, hour_frequency) for m in range(0, 60, minute_frequency) ]


def create_realtime_dataset(location, product, verify=False,validate_with_VIIRS=False,extra_validator_sources = {}):
    site = SiteInfo(location,realtime_site_conf)
    start_time, end_time = site.start_time, site.end_time
    time_dif = end_time - start_time
    log_path = 'logs/failures_' + location + '_' + str(site.start_time) + '_' + str(site.end_time) + '.txt'
    product_band = ''.join(map(lambda item: f"{item['product_name']}{format(item['band'],'02d')}", product))
    goes_tif_dir = RealTimeIncoming_files.replace('$LOC', location).replace('$RESULT_TYPE', goes_folder if validate_with_VIIRS == False else goes_folder+'test')
    site.get_image_dimention()
    # initialize Goes object and prvide file for log
    goes = GoesProcessing(log_path,list(map(lambda item: item['product_name'], product)),list(map(lambda item: item['band'],product))
                                    ,site=site)
    # initialize VIIRS object , this will create firefixel for particular site and define image parameters
    if(validate_with_VIIRS):
        pass
        # not implemented in paper

    
    validate_with_radar = False
    if('radar' in  extra_validator_sources.keys()):
    # if(validate_with_radar):
        validate_with_radar = True
        radarprocessing = extra_validator_sources['radar']
        
    # running for each date
    for i in range(time_dif.days):
        fire_date = str(start_time + datetime.timedelta(days=i))

        if(validate_with_radar):
            unique_time = radarprocessing.get_unique_dateTime(fire_date.replace('-', ''))

        else:
            unique_time = create_dummy_uniquetime()

        # running for ever hhmm for perticular date
        for ac_time in unique_time:
            
            sDATE = dt.datetime.strptime(fire_date + "_" + str(ac_time).zfill(4), '%Y-%m-%d_%H%M')
        
            # now = dt.datetime.utcnow()
            # one_hour_ago = now - dt.timedelta(hours=2)
            # if sDATE < one_hour_ago:
            #     continue
            
            paths = goes.download_goes(fire_date, str(ac_time))
            if -1 not in paths:
                # if(validate_with_VIIRS):
                #     VIIRS_tif_dir = RealTimeIncoming_files.replace('$LOC', location).replace('$RESULT_TYPE', viirs_folder)
                #     v2r_viirs.make_tiff(fire_date, ac_time,VIIRS_tif_dir)
                goes.nc2tiff(fire_date, ac_time, paths, site, site.image_size, goes_tif_dir)
                if verify:
                    GOES_visual_verification(ac_time, fire_date, paths, site, save=False)
