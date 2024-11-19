"""
This script create pojo( object) from config file foe site information like longitude and latitude
and duration of fire

Created on Sun Jul 23 11:17:09 2022

@author: mukul
"""

import yaml

from GlobalValues import site_conf
from timezonefinder import TimezoneFinder

class SiteInfo():

    def __init__(self,location,site_config_file = site_conf):

        self.location = location
        with open(site_config_file, "r", encoding="utf8") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        self.start_time, self.end_time = config.get(location).get('start') , config.get(location).get('end')
        self.latitude , self.longitude = config.get(location).get('latitude') , config.get(location).get('longitude')
        self.rectangular_size = config.get('rectangular_size')
        self.bottom_left = [self.latitude - self.rectangular_size, self.longitude - self.rectangular_size]
        self.top_right = [self.latitude + self.rectangular_size, self.longitude + self.rectangular_size]
        # self.EPSG  = config.get(location).get('EPSG')
        self.EPSG=  self.coordinate2EPSG(self.latitude, self.longitude)
        self.timezone_name = self.get_timeZone()

    def coordinate2EPSG(self,lat,lon):
        start_lon = -126
        end_lon = -60
        start_zone = 32610
        # u.s epsg 32610 to 32620
        # lon_min, lon_max = -125.0, -66.93457 for united states
        for i in range(start_lon, end_lon, 6):
            if i < lon <= i + 6:
                return start_zone
            start_zone +=1
        
    def get_timeZone(self):
        tf = TimezoneFinder()
        timezone_name = tf.timezone_at(lat=self.latitude, lng=self.longitude)
        return timezone_name

    def get_image_dimention(self, res=375,in_crs = 4326):

        from pyproj import Transformer
        self.res = res
        
        # transforming lon lat to utm
        # UTM, Universal Transverse Mercator ( northing and easting)
        # https://www.youtube.com/watch?v=LcVlx4Gur7I

        # https://epsg.io/32611
        # 32610 (126 to 120) ;32611 (120 to 114) ;32612 (114 to 108)

        self.transformer = Transformer.from_crs(in_crs, self.EPSG)
        bottom_left_utm = [int(self.transformer.transform(self.bottom_left[0], self.bottom_left[1])[0]),
                            int(self.transformer.transform(self.bottom_left[0], self.bottom_left[1])[1])]
        top_right_utm = [int(self.transformer.transform(self.top_right[0], self.top_right[1])[0]),
                            int(self.transformer.transform(self.top_right[0], self.top_right[1])[1])]

        top_right_utm = [top_right_utm[0] - (top_right_utm[0] - bottom_left_utm[0]) % self.res,
                            top_right_utm[1] - (top_right_utm[1] - bottom_left_utm[1]) % self.res]
        # adjustment (adding residue) because we want to make equal sized grids on whole area
        # ------
        # ------
        # ------
        # ------
        # creating offset for top right pixel

        lon = [bottom_left_utm[0], top_right_utm[0]]
        lat = [bottom_left_utm[1], top_right_utm[1]]
        # print(lon, lat)

        # setting image parameters
        xmin, ymin, xmax, ymax = [min(lon), min(lat), max(lon), max(lat)]

        # self.transformed_bottom_left= [ymin, xmin]
        # self.transformed_top_right = [ymax, xmax]

        self.transformed_bottom_left= [xmin,ymin]
        self.transformed_top_right = [xmax,ymax]

        nx = round((xmax - xmin) / self.res)
        ny = round((ymax - ymin) / self.res)

        self.image_size = (ny, nx)
