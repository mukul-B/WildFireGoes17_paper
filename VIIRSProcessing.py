"""
This script will
create (per site per date) tiff from VIIRS csv
and download corresponding GOES file and crop it for site bounding box and create TIFF

Created on Sun Jul 23 11:17:09 2022

@author: mukul
"""
import os
import numpy
import numpy as np
import pandas as pd
from osgeo import gdal
from osgeo import osr
from pyproj import Transformer
from scipy.interpolate import griddata
from sklearn.neighbors import NearestNeighbors
from scipy import interpolate

from GlobalValues import VIIRS_tiff_file_name, viirs_dir, VIIRS_OVERWRITE
from os.path import exists as file_exists


class VIIRSProcessing:

    def __init__(self, year="2021", satellite="viirs-snpp", site=None, res=375):


        self.location = site.location
        self.satellite = satellite
        self.crs = site.EPSG
        self.res = res
        self.year = year

        # defining extend of site in lat and lon
        self.bottom_left = site.bottom_left
        self.top_right = site.top_right

        self.transformer = site.transformer
        self.xmin, self.ymin, self.xmax, self.ymax = [site.transformed_bottom_left[0], site.transformed_bottom_left[1], site.transformed_top_right[0], site.transformed_top_right[1]]
        self.nx = site.image_size[1]
        self.ny = site.image_size[0]
        self.image_size = site.image_size
        
    # yearly summaries of VIIRS data is open source avaliable on FIRMS  (https://firms.modaps.eosdis.nasa.gov/country/)
    # After downloading the VIIRS source manually place them in source directory
    def extract_hotspots(self):
        country = 'United_States'
        dtype = np.dtype([
                ('latitude', 'float64'),
                ('longitude', 'float64'),
                ('bright_ti4', 'float64'),
                ('scan', 'float32'),
                ('track', 'float32'),
                ('acq_date', 'U10'),    # String for date
                ('acq_time', 'int32'),     # Assuming time is stored as an integer
                ('satellite', 'object'),   # String for satellite
                ('instrument', 'object'),  # String for instrument
                ('confidence', 'object'),  # String for confidence
                ('version', 'object'),     # String for version
                ('bright_ti5', 'float64'),
                ('frp', 'float32'),
                ('daynight', 'U1'),    # String for day/night
                ('type', 'int32')          # Integer for type
            ])
        source_list = []
        Sdirectory = "VIIRS_Source/" + self.satellite + "_" + self.year + "_" + country + ".csv"
        if os.path.exists(Sdirectory):
            snn_yearly= pd.read_csv(Sdirectory, low_memory=False)
            source_list.append(snn_yearly)

        Sdirectory3 = f'VIIRS_Source_new/fire_archive_SV-C2_{self.year}.csv'
        if os.path.exists(Sdirectory3):
            snpp_pixels = pd.read_csv(Sdirectory3)
            snpp_pixels.rename(columns={'brightness':'bright_ti4'}, inplace=True)
            snpp_pixels.rename(columns={'bright_t31':'bright_ti5'}, inplace=True)
            source_list.append(snpp_pixels)

        Sdirectory2 = f'VIIRS_Source_new/fire_nrt_J1V-C2_{self.year}.csv'
        if os.path.exists(Sdirectory2):
            NOAA_pixels = pd.read_csv(Sdirectory2)
            NOAA_pixels.rename(columns={'brightness':'bright_ti4'}, inplace=True)
            NOAA_pixels.rename(columns={'bright_t31':'bright_ti5'}, inplace=True)
            source_list.append(NOAA_pixels)

        Sdirectory4 = f'VIIRS_Source_new/fire_nrt_SV-C2_{self.year}.csv'
        if os.path.exists(Sdirectory4):
            snpp_nrt_pixels = pd.read_csv(Sdirectory4)
            snpp_nrt_pixels.rename(columns={'brightness':'bright_ti4'}, inplace=True)
            snpp_nrt_pixels.rename(columns={'bright_t31':'bright_ti5'}, inplace=True)
            source_list.append(snpp_nrt_pixels)
         
        VIIRS_pixel = pd.concat(source_list, ignore_index=True)
        # VIIRS_pixel = NOAA_pixels

        self.fire_pixels = VIIRS_pixel

        self.fire_pixels = self.fire_pixels[self.fire_pixels.latitude.gt(self.bottom_left[0])
                                            & self.fire_pixels.latitude.lt(self.top_right[0])
                                            & self.fire_pixels.longitude.gt(self.bottom_left[1])
                                            & self.fire_pixels.longitude.lt(self.top_right[1])]
        
    def get_unique_dateTime(self, fire_date):
        self.fire_data_filter_on_date_and_bbox = self.fire_pixels[self.fire_pixels.acq_date.eq(fire_date)]
        unique_time = self.fire_data_filter_on_date_and_bbox.acq_time.unique()
        return unique_time

    def hhmm_to_minutes(self,hhmm):
        """Convert HHMM format to minutes since midnight."""
        hours = hhmm // 100
        minutes = hhmm % 100
        return hours * 60 + minutes
    
    def get_close_dates(self,unique_time):
        if(len(unique_time) <2):
            return []
        unique_time = np.sort(unique_time)
        unique_time_minutes = np.array([self.hhmm_to_minutes(t) for t in unique_time])
        time_intervel = np.diff(unique_time_minutes)
        min_time_intervel = time_intervel.min()
        short_unique=[]
        for i,j in enumerate(time_intervel):
            if time_intervel[i] < 10:
                short_unique.append(unique_time[i])
                short_unique.append(unique_time[i+1])
        # print(short_unique)
        short_unique = np.unique(short_unique)
        return short_unique
    
    def collapse_close_dates(self,unique_time):
            if(len(unique_time) <2):
                return unique_time
            unique_time = np.sort(unique_time)
            unique_time_minutes = np.array([self.hhmm_to_minutes(t) for t in unique_time])
            time_intervel = np.diff(unique_time_minutes)
            time_intervel = np.append(time_intervel,2400)
            short_unique=[]
            for i,j in enumerate(time_intervel):
                if time_intervel[i] > 10:
                    short_unique.append(unique_time[i])
            return short_unique

    
    # Resample VIIRS samples and write in tiff format
    def make_tiff(self, fire_date,ac_time,viirs_tif_dir = None):

        # output file name
        if viirs_tif_dir is None:
            viirs_tif_dir = viirs_dir.replace('$LOC', self.location) 

        # VIIRS_file_name = self.satellite + '-' + str(fire_date) + "_" + str(ac_time) + '.tif'
        VIIRS_file_name = VIIRS_tiff_file_name.format(fire_date = str(fire_date),
                                                       ac_time = str(ac_time))
        out_file = viirs_tif_dir + VIIRS_file_name

        if ((not VIIRS_OVERWRITE) and file_exists(out_file)):
            return
        # filter firepixel for time of date
        fire_data_filter_on_time = self.fire_data_filter_on_date_and_bbox[
            (self.fire_data_filter_on_date_and_bbox.acq_time.lt(ac_time+1)) & (self.fire_data_filter_on_date_and_bbox.acq_time.gt(ac_time-10))]
        fire_data_filter_on_timestamp = np.array(fire_data_filter_on_time)

        # creating pixel values used in tiff
        b1_pixels,b2_pixels = self.create_raster_array(fire_data_filter_on_timestamp)
        b1_pixels = self.interpolation(b1_pixels)
        # comment normalize_VIIRS for visualizing real values
        # b1_pixels = self.normalize_VIIRS(b1_pixels)
        self.gdal_writter(out_file,[b1_pixels,b2_pixels])

    def create_raster_array(self, fire_data_filter_on_timestamp):
        b1_pixels = np.zeros(self.image_size, dtype=float)
        b2_pixels = np.zeros(self.image_size, dtype=float)
        for k in range(fire_data_filter_on_timestamp.shape[0]):
            record = fire_data_filter_on_timestamp[k]
            # transforming lon lat to utm
            lon_point = self.transformer.transform(record[0], record[1])[0]
            lat_point = self.transformer.transform(record[0], record[1])[1]
            cord_x = round((lon_point - self.xmin) / self.res)
            cord_y = round((lat_point - self.ymin) / self.res)
            if cord_x >= self.nx or cord_y >= self.ny:
                continue
            
            # hanfling Saturated Pixels
            modified_BT = record[2]
            if(record[2] ==208):
                modified_BT = 367
            if(record[2]<record[11] and record[11]<=367):
                modified_BT = record[11]
            # if(modified_BT > 367):
            #     modified_BT = 367

            b1_pixels[-cord_y, cord_x] = max(b1_pixels[-cord_y, cord_x], modified_BT)
            b2_pixels[-cord_y, cord_x] = max(b2_pixels[-cord_y, cord_x], record[12])
        return b1_pixels,b2_pixels

    # check if the zero is farbackground or surronding the fire, used for interpolation
    def nonback_zero(self, b1_pixels, ii, jj):
        checks = [
            (ii - 1, jj - 1), (ii, jj - 1), (ii + 1, jj - 1),
            (ii - 1, jj), (ii + 1, jj),
            (ii - 1, jj + 1), (ii, jj + 1), (ii + 1, jj + 1)
        ]
        for m, n in checks:
            if b1_pixels[m, n] != 0.0:
                return True
        return False

    def interpolation(self, b1_pixels):
        grid_x = np.linspace(self.xmin, self.xmax, self.nx)
        grid_y = np.linspace(self.ymin, self.ymax, self.ny)
        grid_x, grid_y = np.meshgrid(grid_x, grid_y)
        filtered_b12 = []
        bia_x2 = []
        bia_y2 = []
        chec = []
        for ii in range(b1_pixels.shape[0]):
            for jj in range(b1_pixels.shape[1]):
                if ii != 0 and ii != (b1_pixels.shape[0] - 1) and jj != 0 and jj != (b1_pixels.shape[1] - 1):
                    # skiping the border values
                    if b1_pixels[ii, jj] == 0:
                        # interpolation zeros which have at least one surrounding fire pixel
                        if self.nonback_zero(b1_pixels, ii, jj):
                            chec.append((ii, jj))
                            continue
                filtered_b12.append(b1_pixels[ii, jj])
                bia_x2.append(grid_x[ii, jj])
                bia_y2.append(grid_y[ii, jj])
        filtered_b1 = np.array(filtered_b12)
        grid_xx = np.array(bia_x2).reshape(-1, 1)
        grid_yy = np.array(bia_y2).reshape(-1, 1)
        grid = np.hstack((grid_xx, grid_yy))
        grid_z = griddata(grid, filtered_b1, (grid_x, grid_y), method='nearest', fill_value=0)
        # grid_z = griddata(grid, filtered_b1, (grid_x, grid_y), method='linear', fill_value=0)
        # grid_z = griddata(grid, filtered_b1, (grid_x, grid_y), method='cubic', fill_value=0)
        # plot_sample([b1_pixels, grid_z,grid_z_l,grid_z_c], ["Rasterized VIIRS", "nearest","linear","cubic"])

        return grid_z

    def gdal_writter(self, out_file, b_pixels):
        dst_ds = gdal.GetDriverByName('GTiff').Create(
            out_file, self.image_size[1],
            self.image_size[0], len(b_pixels),
            gdal.GDT_Float32)
        # transforms between pixel raster space to projection coordinate space.
        # new_raster.SetGeoTransform((x_min, pixel_size, 0, y_min, 0, pixel_size))
        geotransform = (self.xmin, self.res, 0, self.ymax, 0, -self.res)
        dst_ds.SetGeoTransform(geotransform)  # specify coords
        srs = osr.SpatialReference()  # establish encoding
        srs.ImportFromEPSG(self.crs)  # WGS84 lat/long
        dst_ds.SetProjection(srs.ExportToWkt())  # export coords to file
        for i in range(len(b_pixels)):
            dst_ds.GetRasterBand(i+1).WriteArray(b_pixels[i])
        dst_ds.FlushCache()  # write to disk
        dst_ds = None
