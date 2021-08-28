import pandas as pd
import matplotlib.pyplot as plt
import datetime
import numpy as np
import seaborn as sns

''' Temporal analysis of temperature and ndvi,
instanciation of class needs dict of datas from image collection selected'''

'''minimal code to get informations :
import Smartrees.evo_temp as smet
import Smartrees.date_to_data as smdtd
dict_df=smdtd.Datas().get_data_from_dates()
dat=smet.Temporal(dict_df)
dat.get_evo_allplot();
dat.correlation_plot();
dat.simple_pred_hotspot();
'''

class Temporal() :
    def __init__(self,dict_df):
        self.dict_df=dict_df

    def evo_temp(self, dict_df : dict , column : str = 'NDVI'):
        ''' Create a dataframe with temporal evolution of features for each pixels, as well as the evolution of derivative
        of such features
        Careful as first column of raw and div contains the base value !!! you may want to remove it in later analysis.
        '''

        mmt=pd.DataFrame()
        div_mmt=pd.DataFrame()
        raw_diff_mmt=pd.DataFrame()
        i=0
        for im_id in dict_df:
            date_now=datetime.date(int(im_id.split('_')[-1][:4]),int(im_id.split('_')[-1][4:6]),int(im_id.split('_')[-1][6:]))
            if i==0 :
                raw_diff_mmt[str(date_now)]=dict_df[im_id][column]
                div_mmt[str(date_now)]=dict_df[im_id][column]
            else :
                raw_diff_mmt[str(date_now)]=(dict_df[im_id][column]-mmt[str(date_old)])
                div_mmt[str(date_now)]=(dict_df[im_id][column]-mmt[str(date_old)])/((date_now-date_old).days)
            date_old=date_now
            mmt[str(datetime.date(int(im_id.split('_')[-1][:4]),int(im_id.split('_')[-1][4:6]),int(im_id.split('_')[-1][6:])))]=dict_df[im_id][column]
            i=i+1

        return mmt, div_mmt, raw_diff_mmt

    def get_evo_allfeat(self, normalized = True):
        ''' Gather precisely temperature and ndvi temporal evolution '''
        if normalized :
            temp, div_temp, raw_diff_temp = self.evo_temp(dict_df=self.dict_df,column='Norm_Temp')
        else :
            temp, div_temp, raw_diff_temp = self.evo_temp(dict_df=self.dict_df,column='B10')
        ndvi, div_ndvi, raw_diff_ndvi = self.evo_temp(dict_df=self.dict_df)
        return temp, div_temp, raw_diff_temp, ndvi, div_ndvi, raw_diff_ndvi

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

    def plot_evo_mean(self, data, feature : str = 'NDVI', derivative : bool = False):
        ''' Function plotting evolution of features for given pixel '''
        if derivative :
            plt.plot(data.columns[1:],data.iloc[:,1:].mean());
        else :
            plt.plot(data.columns,data.iloc[:,:].mean());
        plt.xlabel('Dates')
        plt.ylabel(feature)
        plt.xticks(data.columns[::3]);
        plt.title(f'Evolution of the mean {feature} between {data.columns[0]} and {data.columns[-1]}')

    def get_evo_allplot(self, ismean=False, pix=0):
        ''' Function giving a condensed summary of plot for all features, for a given pixel if ismean set to false
        and of the mean value taken over all otherwise
        '''
        if ismean :
            temp, div_temp, raw_diff_temp, ndvi, div_ndvi, raw_diff_ndvi = self.get_evo_allfeat()
            plot = plt.subplots(3,2,figsize=(21,10))
            plt.subplot(3,2,1)
            self.plot_evo_mean(temp,feature='Normalized Temperature')
            plt.subplot(3,2,2)
            self.plot_evo_mean(ndvi)
            plt.subplot(3,2,3)
            self.plot_evo_mean(div_temp,feature='Derivative Norm. Temperature', derivative=True)
            plt.subplot(3,2,4)
            self.plot_evo_mean(div_ndvi, feature='Derivative NDVI', derivative=True)
            plt.subplot(3,2,5)
            self.plot_evo_mean(raw_diff_temp,feature='Raw diff. Norm. Temperature', derivative=True)
            plt.subplot(3,2,6)
            self.plot_evo_mean(raw_diff_ndvi, feature='Raw diff. NDVI', derivative=True)
        else :
            temp, div_temp, raw_diff_temp, ndvi, div_ndvi, raw_diff_ndvi = self.get_evo_allfeat()
            plot = plt.subplots(3,2,figsize=(21,10))
            plt.subplot(3,2,1)
            self.plot_evo(temp,feature='Normalized Temperature', pixel=pix)
            plt.subplot(3,2,2)
            self.plot_evo(ndvi,pixel=pix)
            plt.subplot(3,2,3)
            self.plot_evo(div_temp,feature='Derivative Norm. Temperature', derivative=True, pixel=pix)
            plt.subplot(3,2,4)
            self.plot_evo(div_ndvi, feature='Derivative NDVI', derivative=True, pixel=pix)
            plt.subplot(3,2,5)
            self.plot_evo(raw_diff_temp,feature='Raw diff. Norm. Temperature', derivative=True, pixel=pix)
            plt.subplot(3,2,6)
            self.plot_evo(raw_diff_ndvi, feature='Raw diff. NDVI', derivative=True, pixel=pix)
        return plot

    def correlation_plot(self, div = False):
        ''' Give a correlation of raw_diff_ndvi in function of raw_diff_temp'''
        temp, div_temp, raw_diff_temp, ndvi, div_ndvi, raw_diff_ndvi = self.get_evo_allfeat()
        plot=plt.subplots(figsize=(15,10))
        if div:
            div_temp_all = np.array(div_temp.iloc[:,1:]).reshape(div_temp.shape[0]*(div_temp.shape[1]-1))
            div_ndvi_all = np.array(div_ndvi.iloc[:,1:]).reshape(div_ndvi.shape[0]*(div_ndvi.shape[1]-1))
            sns.scatterplot(div_temp_all,div_ndvi_all);
            plt.xlabel('Norm. Temperature derivative');
            plt.ylabel('NDVI derivative');
            plt.title('Correlation between norm. temperature and ndvi');
        else :
            raw_diff_temp_all = np.array(raw_diff_temp.iloc[:,1:]).reshape(raw_diff_temp.shape[0]*(raw_diff_temp.shape[1]-1))
            raw_diff_ndvi_all = np.array(raw_diff_ndvi.iloc[:,1:]).reshape(raw_diff_ndvi.shape[0]*(raw_diff_ndvi.shape[1]-1))
            sns.scatterplot(raw_diff_temp_all,raw_diff_ndvi_all);
            plt.xlabel('Raw diff. Norm. Temperature');
            plt.ylabel('Raw diff. NDVI');
            plt.title('Correlation between norm. temperature and ndvi');
        return plot

    def simple_pred_hotspot(self, date='2020-08-04'):
        ''' define a combined index for hotspots and print a map image of it'''
        temp, div_temp, raw_diff_temp, ndvi, div_ndvi, raw_diff_ndvi = self.get_evo_allfeat()

        def custom_index_ndvi(x, seuil=0.5):
            if x<0:
                return 0
            elif x>seuil:
                return 0
            return 1-x

        def custom_index_temp(x):
            if x<0:
                return 0
            return x

        index=temp.loc[:,date].map(custom_index_temp)*ndvi.loc[:,date].map(custom_index_ndvi)
        Tree_necessity_index=(index+abs(min(index)))/(max(index)-min(index))*100

        def fill_zeros(data=Tree_necessity_index,size=377*277):
            df=pd.DataFrame(data.copy())
            true_size=len(df)
            df['indice']=df.index
            datframe=pd.DataFrame(np.zeros((size,)),columns={'Tree_index'})
            datframe.loc[df.index,'Tree_index']=df.iloc[:,0]
            return datframe

        Tree_necessity_index_filled =fill_zeros()

        plot=plt.subplots(figsize=(15,10))
        im=np.array(Tree_necessity_index_filled).reshape((377,277))
        plt.imshow(im, cmap='viridis')
        plt.title('Tree necessity index')
        plt.colorbar()
        return plot

def K_to_C(temperature):
    ''' Transform Kelvin into Celsius'''
    return temperature - 273.15
