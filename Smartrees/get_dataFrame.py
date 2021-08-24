import ee
import geehydro
import geemap
from io import StringIO
import folium

import PIL.Image as Im
import numpy as np
import pandas as pd

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
                 corner2=[7.3, 43.75]):
        " Init fonction of class SmarTrees"
        self.ee_image = ee_image
        self.corner1 = corner1
        self.corner2 = corner2
        self.aoi = self.get_aoi()

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
                     scale='30',
                     file_per_band=True):
        " Exports and image with the file name and path in filename"
        geemap.ee_export_image(image,
                               filename=filename,
                               scale=scale,
                               region=self.aoi,
                               file_per_band=True)
        pass
