import ee
import geehydro
import geemap
from io import StringIO
import folium

import PIL.Image as Im
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

"""
La classe SmarTrees est initialisée avec le texte correspondant à l'image earth engine
et  les coins doivent aussi être précisés avec pour valeurs par défaut:
ee_image='LANDSAT/LC08/C01/T1_TOA/LC08_195030_20210729',
corner1=[7.2, 43.65],
corner2=[7.3, 43.75]
"""

# CODE MINIMUM POUR RECUPERER LE DATAFRAME:
"""
data=SmarTrees()
df=data.get_3bands_df()
"""


class SmarTrees():
    def __init__(self,
                 ee_image='LANDSAT/LC08/C01/T1_TOA/LC08_195030_20210729',
                 corner1=[7.2, 43.65],
                 corner2=[7.3, 43.75],scale=30):
        " Init fonction of class SmarTrees"
        self.ee_image = ee_image
        self.corner1 = corner1
        self.corner2 = corner2
        self.aoi = self.get_aoi()
        self.shapes={}
        self.date=ee_image[-8:]
        self.scale=30

    def get_aoi(self):
        "Get The polygon region for ee as a Polygone"
        aoi = ee.Geometry.Polygon([[[self.corner1[0], self.corner1[1]],
                                    [self.corner1[0], self.corner2[1]],
                                    [self.corner2[0], self.corner2[1]],
                                    [self.corner2[0], self.corner1[1]]]], None,
                                  False)
        return aoi

    def get_array_from_image(self, image):
        return geemap.ee_to_numpy(image, region=self.aoi)

    def get_img_band(self, band):
        "GET the band from ee_image"
        img = ee.Image(self.ee_image).select([f'B{band}'])
        return img

    def get_df_band(self, band):
        "GET the datafram from the band from ee_image"
        img = self.get_img_band(band)
        img_arr = self.get_array_from_image(img)
        self.shapes[band]=img_arr.shape
        df = pd.DataFrame(np.concatenate(img_arr), columns=[f'B{band}'])
        return df

    def get_3bands_df(self):
        "GET the datafram from of bands B4,B5,B10 from ee_image"
        df_B4 = self.get_df_band(4)
        df_B5 = self.get_df_band(5)
        df_B10 = self.get_df_band(10)
        df = df_B4.join(df_B5).join(df_B10)
        return df

    def Export_image(self,
                     image,
                     filename='filename',
                     file_per_band=True):
        " Exports and image with the file name and path in filename"
        geemap.ee_export_image(image,
                               filename=filename,
                               scale=self.scale,
                               region=self.aoi,
                               file_per_band=True)
        pass

    def get_pixel_loc(self,index,band=10):
        " Utlisé pour obtenir les coordonnées géographiques des coins du pixel d'indice index"
        # on récupère les données de la carte

        img=self.get_img_band(band)
        img_arr = self.get_array_from_image(img)
        img_arr_shape=img_arr.shape
        index_max=img_arr.shape[0]*img_arr.shape[1]
        print('index_max=',index_max)
        if index >= index_max or index<0 :
            return None
        # countours de la carte
        left_lim=min(self.corner1[0],self.corner2[0])
        right_lim=max(self.corner1[0],self.corner2[0])
        top_lim=max(self.corner1[1],self.corner2[1])
        bot_lim=min(self.corner1[1],self.corner2[1])

        k1=index % img_arr_shape[1]
        k2= index // img_arr_shape[1]
        left_lim_pix=left_lim+(right_lim-left_lim)/img_arr_shape[1]*k1
        right_lim_pix=left_lim+(right_lim-left_lim)/img_arr_shape[1]*(k1+1)
        top_lim_pix=top_lim+(bot_lim-top_lim)/img_arr_shape[0]*(k2)
        bot_lim_pix=top_lim+(bot_lim-top_lim)/img_arr_shape[0]*(k2+1)

        corner1_pix=[left_lim_pix,top_lim_pix]
        corner2_pix=[right_lim_pix,bot_lim_pix]
        output=(corner1_pix,corner2_pix)
        return output

    def output_images(self, df):
        img_B10 = np.array(df['B10'])
        img_NDVI = np.array(df['NDVI'])
        img_B10 = img_B10.reshape((self.shapes[10][0], self.shapes[10][1]))
        img_NDVI = img_NDVI.reshape((self.shapes[4][0], self.shapes[4][1]))
        plt.figure(figsize=(15, 10))
        plt.imshow(img_B10, cmap='coolwarm')
        plt.savefig(f'../output_images/{self.date}_Temp.png')
        plt.close()
        plt.figure(figsize=(15, 10))
        plt.imshow(img_NDVI, cmap='RdYlGn')
        plt.savefig(f'../output_images/{self.date}_NDVI.png')
        plt.close()
        return None

    def get_NDVIandKELVIN(self):

        '''functions that computes NDVI from bands 5 and 4
        (taken from df generated with 'get_3bands_df')
        and returns a dataframe with 2 colonnes, ndvi and kelvin
        '''
        df=self.get_3bands_df()

        b4 = df['B4']
        b5 = df['B5']

        ndvi = (b5 - b4) / (b5 + b4)

        df1 = pd.DataFrame((ndvi), columns=[f'NDVI'])

        df_new = df[['B10']].join(df1)

        return df_new


# Coldpoints and normalization fonctions
    def temperature(self):
        _3bands = self.get_3bands_df()
        return _3bands[["B10"]]

    def check_coldpoints(self,
                         temperature=0,
                         hottest=297):
        temperature = self.temperature()
        temperature.loc[:, 'index'] = temperature.index
        temperature.loc[:, 'value'] = 0
        temperature.loc[temperature[temperature['B10'] < hottest].index,
                        'value'] = 1
        print('0 for hot, 1 for cold')
        return temperature

    def show_coldpoints(self, hottest=297):
        temperatures = self.temperature()
        temperature = self.check_coldpoints(temperatures,
                                            hottest=hottest)
        temp_array = np.array(temperature.value).reshape(377, 277)
        plt.imshow(temp_array, cmap='gray')

    def remove_sea(self):
        #return a df without the sea
        temp = self.check_coldpoints()
        return temp[temp['value'] == 0]

    def z_temperature(self, keepnan=False):
        #return a Dataframe of z_score for temperatures without sea and NDVI
        NDVIandTemperature = self.get_NDVIandKELVIN()
        NDVIandZ = NDVIandTemperature.drop(columns='B10')
        temper = self.remove_sea().B10
        print("wait 2min")
        z = [(temp - np.mean(temper)) / np.std(temper)
             for temp in temper.values]
        z = pd.DataFrame(z, columns=['z_temperatures'])
        if keepnan == True:
            return NDVIandZ.join(z)
        return NDVIandZ.join(z).dropna()
