import ee
import geemap
import geehydro
import folium
import pandas as pd
import datetime
from datetime import timedelta

ee.Initialize()

def get_meta_data(date_start : str = '2017-01-01' , date_stop : str = '2017-01-31', pos : tuple = (7.28045496,43.70684086)):
    ''' Create a collection of images based on a given span of time and a gps position,
    return a dataframe containing the list of images + meta-info (e.g. date, time, cloud cover)
    Format of date : YYYY-MM-DD
    '''
    collection=ee.ImageCollection('LANDSAT/LC08/C01/T1_TOA')\
        .filterDate(f'{date_start}', f'{date_stop}')\
        .filterBounds(ee.Geometry.Point(pos[0],pos[1]))

    arr_id=collection.aggregate_array('system:id').getInfo()
    arr_date=collection.aggregate_array('DATE_ACQUIRED').getInfo()
    arr_time=collection.aggregate_array('SCENE_CENTER_TIME').getInfo()
    arr_sun=collection.aggregate_array('SUN_ELEVATION').getInfo()
    arr_cloud=collection.aggregate_array('CLOUD_COVER').getInfo()

    df=pd.DataFrame()
    df['id']=pd.Series(arr_id)
    df['Date']=pd.Series(arr_date)
    df['Time']=pd.Series(arr_time)
    df['Sun']=pd.Series(arr_sun)
    df['Cloud']=pd.Series(arr_cloud)

    return df

def cloud_out(dataframe, perc = 20.0):
    ''' filter out a metadataframe with a max percentage of cloud cover'''
    df = dataframe[dataframe.Cloud<=perc]
    return df

def mapper(img_id:str, pos = [43.70684086,7.28045496] ):
    ''' produce a folium map with a given image id'''
    mapp = folium.Map(location = pos, zoom_start=10)
    trueColor432=ee.Image(img_id).select(['B4','B3','B2'])
    trueColor432Vis = {'min': 0.0,'max':0.4,}
    mapp.addLayer(trueColor432, trueColor432Vis, 'True Color (432)')
    return mapp

def closest_image(date : str = '22/01/2017', formatDate : int = 0,  pos : tuple = (7.28045496,43.70684086)):
    ''' take a date and return the closest date (in a datetime i.e. computable format) that has an sat. image corresponding to the position given.
    2 format of date possible :
        - 0 is the classical everyday format DD/MM/YYYY
        - 1 is the format used by EarthEngine YYYY-MM-DD

        example of use with query :
            df = get_meta_data(closest_image('22/01/2017'),closest_image('22/01/2017')+timedelta(days=1))
            mapper(df['id'][0])
    '''

    if formatDate == 0:
        date_list = date.split('/')
        dateform = datetime.date(int(date_list[2]),int(date_list[1]),int(date_list[0]))
    elif formatDate == 1 :
        date_list = date.split('-')
        dateform = datetime.date(int(date_list[0]),int(date_list[1]),int(date_list[2]))
    else :
        print('Invalid format of date : 0 -> DD/MM/YYYY or 1 -> YYYY-MM-DD')
        return None
    date_start = dateform-timedelta(days=15)
    date_stop = dateform+timedelta(days=15)
    collection = ee.ImageCollection('LANDSAT/LC08/C01/T1_TOA').filterDate(f'{date_start}', f'{date_stop}').filterBounds(ee.Geometry.Point(pos[0],pos[1]))
    arr_date = collection.aggregate_array('DATE_ACQUIRED').getInfo()

    datetime_df=pd.DataFrame(arr_date, columns={'Dates'})

    def transfo_date_datetime(x:str):
        dat = x.split('-')
        dat2 = datetime.date(int(dat[0]),int(dat[1]),int(dat[2]))
        return dat2

    def transfo_date_diff(x:str):
        dat = x.split('-')
        dat2 = datetime.date(int(dat[0]),int(dat[1]),int(dat[2]))
        return abs(dat2-dateform)

    datetime_df['diff']=datetime_df.Dates.map(transfo_date_diff )
    datetime_df['datetime_format']=datetime_df.Dates.map(transfo_date_datetime )

    return datetime_df.sort_values(by='diff').datetime_format.iloc[0]
