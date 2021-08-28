import pandas as pd
import matplotlib.pyplot as plt
import datetime
from datetime import timedelta
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

    """ correlation plots and dataframe on more that one image """

    def match_one_year(self,ref):
        """ Do a temporal matching between images with one year interval """
        list_date=[i for i in ref]
        list_date_base=[]
        list_date_plusoneY=[]

        def last_usable_date(list_of_date):
            date=list_of_date[-1]
            date_list = date.split('-')
            dateform = datetime.date(int(date_list[0]),int(date_list[1]),int(date_list[2]))
            result = dateform-timedelta(days=356)
            return result

        def transfo_date_datetime(x:str):
            dat = x.split('-')
            dat2 = datetime.date(int(dat[0]),int(dat[1]),int(dat[2]))
            return dat2

        dataf=pd.DataFrame()
        lud=last_usable_date(list_date)
        for date in list_date:
            date_list = date.split('-')
            dateform = datetime.date(int(date_list[0]),int(date_list[1]),int(date_list[2]))
            if dateform<lud:
                list_date_base.append(dateform)
                list_date_plusoneY.append(dateform+timedelta(days=365))
        list_date=[transfo_date_datetime(x) for x in list_date]
        dataf['list']=list_date
        for i in list_date_plusoneY:
            dataf[str(i)]=abs(dataf.list-i)
        corresp=[]
        for i in dataf.columns[1:]:
            corresp.append(dataf[['list',i]].sort_values(by=i).list.iloc[0])
        match_table=pd.DataFrame(np.array([list_date_base,corresp]).T,columns=['base','corresp'])
        return list_date_base,corresp,match_table

    def interval_diff(self, base, corresp, feat):
        """ compute substraction between two sets of images at a given interval """
        df = pd.DataFrame()

        for i,j in zip(base,corresp):
            df = pd.concat((df,feat[str(j)]-feat[str(i)]),axis=0)

        return df

    def correlation_plot_all(self):
        ''' plot correlations between ndvi and temp between two sets of images with a given temporal interval '''
        temp, div_temp, raw_diff_temp, ndvi, div_ndvi, raw_diff_ndvi = self.get_evo_allfeat()
        base,corresp,match_table=self.match_one_year(temp.columns)

        plot=plt.subplots(figsize=(15,10))
        raw_diff_temp_all = np.array(self.interval_diff(base, corresp, temp).iloc[:,0])
        raw_diff_ndvi_all = np.array(self.interval_diff(base, corresp, ndvi).iloc[:,0])
        sns.scatterplot(raw_diff_temp_all,raw_diff_ndvi_all);
        plt.xlabel('Raw diff. Norm. Temperature');
        plt.ylabel('Raw diff. NDVI');
        plt.title('Correlation between norm. temperature and ndvi');
        return plot

    def correlation_plot_all_sequential(self):
        """Variant of previous function giving """
        temp, div_temp, raw_diff_temp, ndvi, div_ndvi, raw_diff_ndvi = self.get_evo_allfeat()
        base,corresp,match_table=self.match_one_year(temp.columns)

        plot=plt.subplots(figsize=(15,10))
        for i,j in zip(base,corresp):
            raw_diff_temp_all = temp[str(j)]-temp[str(i)]
            raw_diff_ndvi_all = ndvi[str(j)]-ndvi[str(i)]
            sns.scatterplot(raw_diff_temp_all,raw_diff_ndvi_all);
            plt.xlabel('Raw diff. Norm. Temperature');
            plt.ylabel('Raw diff. NDVI');
            plt.title('Correlation between norm. temperature and ndvi');
        return plot

    def unite_oneY(self):
        temp, div_temp, raw_diff_temp, ndvi, div_ndvi, raw_diff_ndvi = self.get_evo_allfeat()
        base,corresp,match_table=self.match_one_year(temp.columns)

        raw_diff_temp_all = np.array(self.interval_diff(base, corresp, temp).iloc[:,0])
        raw_diff_ndvi_all = np.array(self.interval_diff(base, corresp, ndvi).iloc[:,0])
        df=pd.concat((pd.Series(raw_diff_temp_all),pd.Series(raw_diff_ndvi_all)))
        return df


    """ predictions with specific index"""

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
