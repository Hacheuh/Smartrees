from Smartrees import get_dataFrame
import ee
import numpy as np
#import geehydro
#import geemap
#from io import StringIO
#import folium

#import PIL.Image as Im
#import pandas as pd
import matplotlib.pyplot as plt

'''These fonctions are here to visualize hot points and cold points by default
in Nice, remove extremes values of temperature, for exemple in Nice in the sea
and to normalize temperatures '''

def temperature():
    ee.Initialize()
    data = get_dataFrame.SmarTrees()
    _3bands = data.get_3bands_df()
    return _3bands[["B10"]]


def check_coldpoints(all_temperatures=temperature(), hottest=297):
    temperature = all_temperatures
    temperature.loc[:, 'index'] = temperature.index
    temperature.loc[:, 'value'] = 0
    temperature.loc[temperature[temperature['B10'] < hottest].index,'value'] = 1
    print('0 for hot, 1 for cold')
    return temperature


def show_coldpoints(all_temperatures=temperature(), hottest=297):
    temperature = check_coldpoints(all_temperatures=all_temperatures,hottest=hottest)
    temp_array = np.array(temperature.value).reshape(377, 277)
    plt.imshow(temp_array, cmap='gray')


def remove_sea():
    #return a df without the sea
    temp = check_coldpoints()
    return temp[temp['value'] == 0]


def z_temperature():
    #return a list of z_score for temperatures without sea
    temper = remove_sea().B10
    print("wait 2min")
    z = [(temp - np.mean(temper)) / np.std(temper) for temp in temper.values]
    return z
