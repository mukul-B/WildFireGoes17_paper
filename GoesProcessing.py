"""
This script contains GoesProcessing class contain functionality to
to download GOES reference_data for date , product band and mode
and resample it to tif file for particular site

Created on Sun Jul 23 11:17:09 2022

@author: mukul
"""
import datetime as dt
import warnings
from os.path import exists as file_exists

import numpy as np
import s3fs
from cartopy.io.img_tiles import GoogleTiles
from matplotlib import pyplot as plt
from pyproj import Transformer
from pyresample.geometry import AreaDefinition
from satpy import Scene

from GlobalValues import RAD, FDC, GOES_ndf, GOES_OVERWRITE, GOES_tiff_file_name
import cartopy.crs as ccrs
from osgeo import gdal
from osgeo import osr


warnings.filterwarnings('ignore')


class GoesProcessing:
    def __init__(self, log_path, product_name, bands, site):
        # failure log ex missing files logged in file
        self.product_name = None
        self.band = None
        self.g_reader = None
        self.failures = open(log_path, 'w')
        # self.band = band
        self.band = [('C' + str(band).zfill(2) if band else "") for band in bands]
        self.product_name = product_name
        self.site = site

        # product to g_reader for Satpy
        self.fs = s3fs.S3FileSystem(anon=True)
        self.g_reader = []
        for pn in product_name:
            g_reader = pn.split('-')
            g_reader = '_'.join(g_reader[:2]).lower()
            g_reader = 'abi_l2_nc' if (g_reader == 'abi_l2') else g_reader
            self.g_reader.append(g_reader)
        
        self.area_def = self.get_areaDefination_old(site)
        # area_def_old = self.get_areaDefination_old(site)

        # if(self.area_def!=area_def_old):
        #     print(self.area_def,area_def_old)
        # return 


    def __del__(self):
        self.failures.close()
    
    def set_image_transformerAndsize(self, site):
        
        self.transformer = site.transformer
        self.xmin, self.ymin, self.xmax, self.ymax = [site.transformed_bottom_left[0], site.transformed_bottom_left[1], site.transformed_top_right[0], site.transformed_top_right[1]]
        self.nx = site.image_size[1]
        self.ny = site.image_size[0]
        self.image_size = site.image_size
    

    def download_goes(self, fire_date, ac_time):

        sDATE = dt.datetime.strptime(fire_date + "_" + ac_time.zfill(4), '%Y-%m-%d_%H%M')
        
        # no more GOES -17 after 10 jan 2023 4 p.m
        # GOES 18 has issue from 9/8/2022 to 10/13/2022 ( use goes -17 for this range)
        G18_overlap_start_date = dt.datetime(2022, 9, 1)
        # G18_start_date = dt.datetime(2022, 10, 14)
        G18_start_date = dt.datetime(2022, 11, 15)

        # Compare the dates
        if sDATE > G18_start_date:
            bucket_names=['noaa-goes18']
        elif sDATE > G18_overlap_start_date:
            bucket_names = ['noaa-goes18','noaa-goes17']
        else:
            bucket_names=['noaa-goes17']
        if(self.site.longitude > -109):
            bucket_names=['noaa-goes16']

        for bucket_name in bucket_names:
            return_paths =  [self.download_goes_product(sDATE, product_name, band,bucket_name=bucket_name) for product_name , band in zip(self.product_name,self.band)]
            if -1  not in return_paths:
                path = [self.download_paths(first_file) for first_file in return_paths]
                # path = [-1,-1,-1]
                # self.validate_same_GOESBanddates(sDATE, return_paths)
                
                return path
            
        return [-1,-1,-1]

    def validate_same_GOESBanddates(self, sDATE, return_paths):
        date_prefix = [ first_file.split("_")[3] for first_file in return_paths]
        date_same = np.all([ dater == date_prefix[0] for dater in date_prefix])
                
        if(date_same == False):
            print('error not all date same')
        ret_date = dt.datetime.strptime(date_prefix[0], 's%Y%j%H%M%S%f')
        ret_date_diff = abs((sDATE - ret_date).total_seconds() // 60)
        if(ret_date_diff > 3):
            print('error diff',ret_date_diff)
        
        
    # download GOES reference_data for date , product band and mode , finction have default values
    def download_goes_product(self, sDATE, product_name, band, mode='M6',
                      bucket_name='noaa-goes17'):

        # extract date parameter of AWS request from given date and time
        setelite_file_prefix = bucket_name.replace('noaa-goes','G')
        
        minute = sDATE.minute
        if(minute == 59):
            sDATE = sDATE + dt.timedelta(minutes=1)

        day_of_year = sDATE.timetuple().tm_yday
        year = sDATE.year
        hour = sDATE.hour
        minute = sDATE.minute

        # fdc does have commutative bands
        band = "" if (self.product_name == FDC) else band
        # Write prefix for the files of inteest, and list all files beginning with this prefix.
        prefix = f'{bucket_name}/{product_name}/{year}/{day_of_year:03.0f}/{hour:02.0f}/'
        file_prefix = f'OR_{product_name}-{mode}{band}_{setelite_file_prefix}_s{year}{day_of_year:03.0f}{hour:02.0f}'

        # print(prefix,file_prefix)
        # listing all files for product date for perticular hour
        files = self.fs.ls(prefix)

        # filtering band and mode from file list
        files = [i for i in files if file_prefix in i]

        # return if file not present for criteria and write in log
        if len(files) == 0:
            # print("fine not found in GOES",sDATE )
            self.failures.write("No Match found for {}\n".format(sDATE))
            return -1

        # find closed goes fire from viirs( the closest minutes)
        last, closest = 0, 0
        found = 0
        for index, file in enumerate(files):
            fname = file.split("/")[-1]
            splits = fname.split("_")
            g_time = dt.datetime.strptime(splits[3], 's%Y%j%H%M%S%f')
            if int(dt.datetime.strftime(g_time, '%M')) < int(minute):
                last = index
            if abs(int(dt.datetime.strftime(g_time, '%M')) - int(minute)) < 3:
                closest = index
                found = 1
                break
        
        # return if file not present for criteria and write in log
        if found == 0:
            # print("fine not found in GOES",sDATE , dt.datetime.now(dt.timezone.utc))
            self.failures.write("No Match found for {}\n".format(sDATE))
            return -1
        
        # downloading closed file
        first_file = files[closest]
        return first_file

    def download_paths(self, first_file):
        # out_file=  "GOES-"+str(fire_date)+"_"+str(ac_time)+".nc"
        out_file = first_file.split('/')[-1]
        # path = directory + '/' + product_name + "/" + out_file
        # print(path)
        path = GOES_ndf + "/" + out_file
        if not file_exists(path):
            print('\nDownloading files from AWS...', end='', flush=True)
            self.fs.download(first_file, path)
            print(first_file, "completed")
        return path

    def showgoes(self, lon, lat, data, bbox):
        proj = ccrs.PlateCarree()

        class StreetmapESRI(GoogleTiles):
            # shaded relief
            def _image_url(self, tile):
                x, y, z = tile
                url = ('https://server.arcgisonline.com/ArcGIS/rest/services/' \
                       'World_Street_Map/MapServer/tile/{z}/{y}/{x}.jpg').format(
                    z=z, y=y, x=x)
                return url
        plt.figure(figsize=(6, 3))
        ax = plt.axes(projection=proj)
        bbox2 = [bbox[0], bbox[2], bbox[1], bbox[3]]
        # ax.add_image(StreetmapESRI(), 10)
        # ax.set_extent(bbox)
        # print(bbox)
        cmap = plt.colormaps['YlOrRd']
        transformer = Transformer.from_crs(32611, 4326)
        proj = []
        for ln in lon.values:
            for lt in lat.values:
                proj.append(transformer.transform(ln, lt))

        # proj = [transformer.transform(x, y) for x, y in zip(lon.values,lat.values)]
        lon = np.array([p[0]  for p in proj]).reshape(data.shape)
        lat = np.array([p[1]  for p in proj]).reshape(data.shape)

        p = ax.pcolormesh(lon, lat, data,
                          transform=ccrs.PlateCarree(),
                          alpha=0.9)

        # cbar = plt.colorbar(p, shrink=0.5)
        gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=0, alpha=0.5)
        gl.top_labels = False
        gl.right_labels = False
        gl.xlines = False
        gl.ylines = False
        plt.show()
        print('Done.')

        plt.close('all')

    #   resampling GOES file for given site and writing in a tiff file
    def nc2tiff(self, fire_date, ac_time, paths, site, image_size, directory):


        # path = paths[0]

        # ouput file location
        # out_file = "GOES-" + str(fire_date) + "_" + str(ac_time) + '.tif'
        out_file = GOES_tiff_file_name.format(fire_date = str(fire_date),
                                              ac_time = str(ac_time))
        out_path = directory + out_file

        if ((not GOES_OVERWRITE) and file_exists(out_path)):
            return

        # creating bouding box from site information
        band = self.band
        # band = ('C' + str(band).zfill(2) if band else "")
        layer = "Mask" if (self.product_name == FDC) else band
        latitude, longitude = site.latitude, site.longitude
        rectangular_size = site.rectangular_size
        EPSG = site.EPSG

        # using satpy to crop goes for the given site
        try:
            goes_scene = Scene(reader=self.g_reader,
                               filenames=paths)
        except:
            self.failures.write("issue in satpy netcdf read {}\n".format(paths))
            return -1
        goes_scene.load(layer)
        goes_scene = goes_scene.resample(self.area_def)

        out_val = [np.nan_to_num(goes_scene[band].values) for band in layer]

        self.gdal_writter(out_path,EPSG,image_size,out_val)
    
    def gdal_writter(self, out_file, crs, image_size, b_pixels):
        dst_ds = gdal.GetDriverByName('GTiff').Create(
            out_file, image_size[1],
            image_size[0], len(b_pixels),
            gdal.GDT_Float32)
        # transforms between pixel raster space to projection coordinate space.
        # new_raster.SetGeoTransform((x_min, pixel_size, 0, y_min, 0, pixel_size))
        geotransform = (self.xmin, 375, 0, self.ymax, 0, -375)
        dst_ds.SetGeoTransform(geotransform)  # specify coords
        srs = osr.SpatialReference()  # establish encoding
        srs.ImportFromEPSG(crs)  # WGS84 lat/long
        dst_ds.SetProjection(srs.ExportToWkt())  # export coords to file
        for i in range(len(b_pixels)):
            dst_ds.GetRasterBand(i+1).WriteArray(b_pixels[i]) 
        # dst_ds.GetRasterBand(2).WriteArray(b1_pixels[1])
        dst_ds.FlushCache()  # write to disk
        dst_ds = None
        
    def get_areaDefination(self, site):
        # defining area definition with image size , projection and extend
        EPSG, image_size = site.EPSG, site.image_size
        area_id = 'given'
        description = 'given'
        proj_id = 'given'
        projection = 'EPSG:' + str(EPSG)
        width = image_size[1]
        height = image_size[0]
        self.xmin, self.ymin, self.xmax, self.ymax = [site.transformed_bottom_left[0], site.transformed_bottom_left[1], site.transformed_top_right[0], site.transformed_top_right[1]]
        
        # the lat lon is changed when using utm !?
        area_extent = (site.transformed_bottom_left[0], site.transformed_bottom_left[1], site.transformed_top_right[0], site.transformed_top_right[1])
        area_def = AreaDefinition(area_id, description, proj_id, projection,
                                  width, height, area_extent)
        return area_def

    # get area defination for satpy, with new projection and bounding pox
    def get_areaDefination_old(self, site):


        latitude, longitude = site.latitude, site.longitude
        rectangular_size = site.rectangular_size
        EPSG = site.EPSG
        image_size = site.image_size

        bottom_left = [latitude - rectangular_size, longitude - rectangular_size]
        top_right = [latitude + rectangular_size, longitude + rectangular_size]

        # transforming bounding box coordinates for new projection
        transformer = Transformer.from_crs(4326, EPSG)
        bottom_left_utm = [int(transformer.transform(bottom_left[0], bottom_left[1])[0]),
                           int(transformer.transform(bottom_left[0], bottom_left[1])[1])]
        top_right_utm = [int(transformer.transform(top_right[0], top_right[1])[0]),
                         int(transformer.transform(top_right[0], top_right[1])[1])]
        #TODO: need to check  this, the login needs to be added for consistency but, the UNET evaluation proceduce poor results with this

        # top_right_utm = [top_right_utm[0] - (top_right_utm[0] - bottom_left_utm[0]) % 375,
        #                     top_right_utm[1] - (top_right_utm[1] - bottom_left_utm[1]) % 375]
        
        lon = [bottom_left_utm[0], top_right_utm[0]]
        lat = [bottom_left_utm[1], top_right_utm[1]]

        # setting image parameters
        self.xmin, self.ymin, self.xmax, self.ymax = [min(lon), min(lat), max(lon), max(lat)]

        # defining area definition with image size , projection and extend
        area_id = 'given'
        description = 'given'
        proj_id = 'given'
        projection = 'EPSG:' + str(EPSG)
        width = image_size[1]
        height = image_size[0]

        # the lat lon is changed when using utm !?
        area_extent = (bottom_left_utm[0], bottom_left_utm[1], top_right_utm[0], top_right_utm[1])
        area_def = AreaDefinition(area_id, description, proj_id, projection,
                                  width, height, area_extent)
        return area_def
