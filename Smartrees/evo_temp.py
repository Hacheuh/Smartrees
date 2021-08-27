import pandas as pd
import matplotlib.pyplot as plt
import datetime

''' Temporal analysis of temperature and ndvi,
instanciation of class needs dict of datas from image collection selected'''

class Temporal() :
    def __init__(self,dict_df):
        self.dict_df=dict_df

    def evo_temp(self, dict_df : dict , column : str = 'NDVI'):
        ''' Create a dataframe with temporal evolution of features for each pixels, as well as the evolution of derivative
        of such features'''
        mmt=pd.DataFrame()
        div_mmt=pd.DataFrame()
        i=0
        for im_id in dict_df:
            date_now=datetime.date(int(im_id.split('_')[-1][:4]),int(im_id.split('_')[-1][4:6]),int(im_id.split('_')[-1][6:]))
            if i==0 :
                div_mmt[str(date_now)]=dict_df[im_id][column]
            else :
                div_mmt[str(date_now)]=(dict_df[im_id][column]-mmt[str(date_old)])/((date_now-date_old).days)
            date_old=date_now
            mmt[str(datetime.date(int(im_id.split('_')[-1][:4]),int(im_id.split('_')[-1][4:6]),int(im_id.split('_')[-1][6:])))]=dict_df[im_id][column]
            i=i+1

        return mmt, div_mmt

    def get_evo_allfeat(self):
        ''' Gather precisely temperature and ndvi temporal evolution '''
        temp, div_temp = self.evo_temp(dict_df=self.dict_df,column='B10')
        ndvi, div_ndvi = self.evo_temp(dict_df=self.dict_df)
        return temp, div_temp, ndvi, div_ndvi

    def plot_evo(self, data, feature : str = 'NDVI', pixel : int = 0, derivative : bool = False):
        ''' Function plotting evolution of features for given pixel '''
        if derivative :
            plt.plot(data.columns[1:],data.iloc[pixel,1:]);
        else :
            plt.plot(data.columns,data.iloc[pixel,:]);
        plt.xlabel('Dates')
        plt.ylabel(feature)
        plt.xticks(data.columns[::3]);
        plt.title(f'Evolution of {feature} between {data.columns[0]} and {data.columns[-1]}')

    def get_evo_allplot(self):
        ''' Function giving a condensed summary of plot for all features, for a given pixel '''
        temp, div_temp, ndvi, div_ndvi = self.get_evo_allfeat()
        plot = plt.subplots(2,2,figsize=(15,10))
        plt.subplot(2,2,1)
        self.plot_evo(K_to_C(temp),feature='Temperature (°C)')
        plt.subplot(2,2,2)
        self.plot_evo(ndvi)
        plt.subplot(2,2,3)
        self.plot_evo(div_temp,feature='Derivative Temperature (°C.)', derivative=True)
        plt.subplot(2,2,4)
        self.plot_evo(div_ndvi, feature='Derivative NDVI', derivative=True)
        return plot


def K_to_C(temperature):
    ''' Transform Kelvin into Celsius'''
    return temperature - 273.15
