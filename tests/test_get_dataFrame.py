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
from Smartrees import get_dataFrame

ee.Initialize()
res = get_dataFrame.SmarTrees()
def test_base_init():
    assert len(res.corner2)==2
    assert len(res.corner1)==2
    assert len(res.pos)==2
    assert type(res.scale) == int

def test_get_3bands_df():
    #res = get_dataFrame.SmarTrees()
    df = res.get_3bands_df()
    assert len(list(df.columns))==3

def test_get_NDVIandKelvin():
    #res = get_dataFrame.SmarTrees()
    df = res.get_NDVIandKELVIN()
    assert max(df.NDVI)<=1
    assert min(df.B10)>0

def test_display_folium_map():
    #res = get_dataFrame.SmarTrees()
    maps = res.display_folium_map()
    assert type(maps) == folium.folium.Map

def test_z_temperature():
    #data = get_dataFrame.SmarTrees()
    assert list(res.z_temperature().columns) == ['Norm_Temp', 'NDVI']
    assert np.unique(res.z_temperature()['Norm_Temp']<10)== True
