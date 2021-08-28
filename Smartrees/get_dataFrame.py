import ee
import geehydro
import geemap
from io import StringIO
import folium

import PIL.Image as Im
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import folium.plugins
from folium import raster_layers
import branca
import branca.colormap as cmp
"""
La classe SmarTrees est initialisée avec le texte correspondant à l'image earth engine
et  les coins doivent aussi être précisés avec pour valeurs par défaut:


ee_image='LANDSAT/LC08/C01/T1_TOA/LC08_195030_20210729',     Le nom de l'image earth engine
corner1=[7.2, 43.65],     ---->     Les coins utilisés pour la reconstruction des images
corner2=[7.3, 43.75],

scale=30              -------->     échelle pour la prise des images
sea_pixels            -------->     Contient None ou une séries avec pour chaque indice des
                                    pixels des images 1 si c'est de la terre ou 0 pour de l'eau
sea_filtering          ------->     est ce que la mer est filtrée


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
                 corner2=[7.3, 43.75],
                 scale=30,
                 sea_pixels=None,
                 sea_filtering=1):
        " Init fonction of class SmarTrees"
        self.ee_image = ee_image
        self.corner1 = corner1
        self.corner2 = corner2
        self.aoi = self.get_aoi()           # Region polygone utilisée par earth engine
        self.shapes = {}                    # dictionnaire de dimensions des images (la clé correspond à la bande utilisée sur l'image ee)
        self.date = ee_image[-8:]           # Date récupérée d'après le nom du fichier
        self.scale = 30
        self.sea_filtering = sea_filtering

        self.sea_pixels = sea_pixels

    def get_aoi(self):
        "Get The polygon region for ee as a Polygone"
        aoi = ee.Geometry.Polygon([[[self.corner1[0], self.corner1[1]],
                                    [self.corner1[0], self.corner2[1]],
                                    [self.corner2[0], self.corner2[1]],
                                    [self.corner2[0], self.corner1[1]]]], None,
                                  False)
        return aoi

    def get_array_from_image(self, image):
        """ Transforms EE image in numpy Array based on the aoi region """
        return geemap.ee_to_numpy(image, region=self.aoi)

    def get_img_band(self, band):
        "GETs the band from ee_image"
        img = ee.Image(self.ee_image).select([f'B{band}'])
        return img

    def get_df_band(self, band):
        "GETs the datafram from the band from ee_image"
        img = self.get_img_band(band)
        img_arr = self.get_array_from_image(img)
        self.shapes[band] = img_arr.shape
        df = pd.DataFrame(np.concatenate(img_arr), columns=[f'B{band}'])
        return df

    def get_3bands_df(self):
        "GETs the datafram from of bands B4,B5,B10 from ee_image"
        df_B4 = self.get_df_band(4)
        df_B5 = self.get_df_band(5)
        df_B10 = self.get_df_band(10)
        df = df_B4.join(df_B5).join(df_B10)
        return df

    def Export_image(self, image, filename='filename', file_per_band=True):
        " Exports and image with the file name and path in filename"
        geemap.ee_export_image(image,
                               filename=filename,
                               scale=self.scale,
                               region=self.aoi,
                               file_per_band=True)
        pass

    def get_pixel_loc(self, index, band=10):
        " Utlisé pour obtenir les coordonnées géographiques des coins du pixel d'indice index"
        # on récupère les données de la carte

        img = self.get_img_band(band)
        img_arr = self.get_array_from_image(img)
        img_arr_shape = img_arr.shape
        index_max = img_arr.shape[0] * img_arr.shape[1]
        print('index_max=', index_max)
        if index >= index_max or index < 0:
            return None
        # countours de la carte
        left_lim = min(self.corner1[0], self.corner2[0])
        right_lim = max(self.corner1[0], self.corner2[0])
        top_lim = max(self.corner1[1], self.corner2[1])
        bot_lim = min(self.corner1[1], self.corner2[1])

        k1 = index % img_arr_shape[1]
        k2 = index // img_arr_shape[1]
        left_lim_pix = left_lim + (right_lim -
                                   left_lim) / img_arr_shape[1] * k1
        right_lim_pix = left_lim + (right_lim -
                                    left_lim) / img_arr_shape[1] * (k1 + 1)
        top_lim_pix = top_lim + (bot_lim - top_lim) / img_arr_shape[0] * (k2)
        bot_lim_pix = top_lim + (bot_lim - top_lim) / img_arr_shape[0] * (k2 +
                                                                          1)

        corner1_pix = [left_lim_pix, top_lim_pix]
        corner2_pix = [right_lim_pix, bot_lim_pix]
        output = (corner1_pix, corner2_pix)
        return output

    def output_images(self, df):
        """ Save NDVI and B10 images in output_images """
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
        df = self.get_3bands_df()
        b4 = df['B4']
        b5 = df['B5']
        ndvi = (b5 - b4) / (b5 + b4)
        df1 = pd.DataFrame((ndvi), columns=[f'NDVI'])
        df_new = df[['B10']].join(df1)

        return df_new


# Display map folium of temperature and NDVI

    def display_folium_map(self, min_temp=20, max_temp=40):
        """ Displays folium map of Temp (Celsius) and NDVI with scales"""
        linearndvi = cmp.LinearColormap(
            ['#d73027', '#fc8d59', '#fee08b', '#d9ef8b', '#91cf60', '#1a9850'],
            vmin=-1,
            vmax=1,
            caption='NDVI - Vegetation index'  #Caption for Color scale or Legend
        )

        palettetemp = ['blue', '#fddbc7', 'red']
        linear_temp = cmp.LinearColormap(
            palettetemp,
            vmin=min_temp,
            vmax=max_temp,
            caption='Temperature (°C)'  #Caption for Color scale or Legend
        )

        image = ee.Image(self.ee_image)
        nir = image.select('B5')
        red = image.select('B4')
        b10 = image.select('B10')
        ndvi = nir.subtract(red).divide(nir.add(red)).rename('NDVI')
        temp = b10.subtract(273.15)

        self.pos = [(corner1[0] + corner2[0]) / 2,
                    (corner1[1] + corner2[1]) / 2]

        mapNice = folium.Map(location=[self.pos[1], self.pos[0]],
                             zoom_start=12)
        mapNice.addLayer(temp, {
            'min': min_temp,
            'max': max_temp,
            'palette': palettetemp
        }, 'Temp')
        mapNice.addLayer(
            ndvi, {
                'palette': [
                    '#d73027', '#fc8d59', '#fee08b', '#d9ef8b', '#91cf60',
                    '#1a9850'
                ]
            }, 'NDVI')
        #FloatImage(image_ndvi, bottom=0, left=10).add_to(mapNice)

        folium.LayerControl().add_to(mapNice)
        mapNice.add_child(linearndvi)
        mapNice.add_child(linear_temp)
        mapNice
        #linear_temp,linearndvi
        return mapNice

    # Coldpoints and normalization fonctions
    def temperature(self):
        _3bands = self.get_3bands_df()
        return _3bands[["B10"]]

    def remove_sea(self, working_df):
        """ REMOVES SEA PIXELS OF A DATAFRAME based on the sea_pixel_table output of sea_pixel_of_Nice_ref_image class method"""

        if type(self.sea_pixels) == "<class 'pandas.core.frame.DataFrame'>":
            print(
                'No criteria for exclusion of the sea, plz provide a sea_pixels df when instanciating class Smartrees'
            )
            return working_df
        output_df = working_df.join(self.sea_pixels)
        output_df.columns = ['Norm_Temp', 'NDVI', 'sea_pixel']
        return output_df[output_df['sea_pixel'] == 1][['Norm_Temp', 'NDVI']]

    def z_temperature(self, keepnan=False):

        temper = self.get_NDVIandKELVIN()
        temper.columns = ['Norm_Temp', 'NDVI']
        if self.sea_filtering == 1:

            temper = self.remove_sea(temper)
        means = np.mean(temper.Norm_Temp)
        stds = np.std(temper.Norm_Temp)

        def z_score(x, means=means, stds=stds):
            return (x - means) / stds

        temper['Norm_Temp'] = temper['Norm_Temp'].map(z_score)
        temper.columns = ['Norm_Temp', 'NDVI']
        if keepnan == True:
            return temper
        return temper.dropna()
