"""
This script contains RadarProcessing class that  load geojason files and plot them

Created on Sun june 23 11:17:09 2023

@author: mukul
"""
import json
import os
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')


class RadarProcessing:
    def __init__(self,location):
        self.d = 1
        self.dir = f'radar_data/{location}'
        if(location=='Bear'):
            self.file_r = 'radar_data/Bear/bear_{date_radar}_smooth_perim.geojson'
        if(location=='Caldor'):
            self.file_r = 'radar_data/Caldor/Caldor_{date_radar}_smooth_perim_new.geojson'
        if(location=='ParkFire'):
            self.file_r = 'radar_data/ParkFire/ParkFire_{date_radar}_smooth_perim.geojson'
            
    
    def get_unique_dateTime(self,date):
        radar_list = os.listdir(self.dir)
        date_list = []
        for v_file in sorted(radar_list):
            if not v_file.startswith('._'):
                try:
                    if v_file.split("_")[1][:8] == date:
                        date_list.append(v_file.split("_")[1][-4:])
                except:
                    continue

        return date_list

    def plot_radar_csv(self, file_r, ax):
        listx = []
        listy = []
        data = pd.read_csv(file_r, header=None)
        for ind in range(data.shape[1]):
            listx.append(data[ind][0])
            listy.append(data[ind][1])
        ax.plot(listx, listy)

    def read_json_perim(self, filename):
        fjson = filename
        try:
            with open(fjson) as f:
                gj_f = json.load(f)['features']
        except Exception as e:
            print(f"An error occurred: {e}")
            return None
        gj = [i for i in gj_f if i['geometry'] is not None]
        if len(gj) == 0:
            return None
        if gj[0]['geometry']['type'] == 'MultiPolygon':
            mpoly = gj[0]['geometry']['coordinates']
            perim = []
            if len(mpoly) == 1:
                mpoly = mpoly[0]
            for kk, ii in enumerate(mpoly):
                #TODO: this work for caldor but commented lines work for park fire 
                perim.append(np.squeeze(np.array(ii)))
                # for kk2, ii2 in enumerate(ii):
                #     perim.append(np.squeeze(np.array(ii2)))
        else:
            perim = []
            for kk, ii in enumerate(gj):
                perim.append(np.squeeze(np.array(ii['geometry']['coordinates'])))
        return perim

    def plot_radar_json(self, date_radar, ax):
        perim = self.read_json_perim(self.file_r.format(date_radar=date_radar))
        if (not perim):
            return None
        for peri_p in perim:
            if not peri_p.shape[0] <= 2:
                # radar_poly = Polygon([(i[0], i[1]) for i in peri_p])
                # ax.fill(*radar_poly.exterior.xy, color='dimgray', transform=ccrs.PlateCarree(), zorder=1, alpha=0.8)
                listx = []
                listy = []
                for ind in range(peri_p.shape[0]):
                    listx.append(peri_p[ind][0])
                    listy.append(peri_p[ind][1])
                ax.plot(listx, listy)
                # ax.plot(listx, listy, linewidth=0.6)
        return perim
