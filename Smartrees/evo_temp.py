import Smartrees.date_to_data as smdtd
import pandas as pd

def evo_temp(dict_df : dict, column : str = 'NDVI'):
    mmt=pd.DataFrame()
    div_mmt=pd.DataFrame(columns={"base"})
    i=0
    for im_id in dict_df:
        if i==0 :
            div_mmt['base']=dict_df[im_id][column]
        elif i==1 :
            div_mmt['n=1']=dict_df[im_id][column]-div_mmt['base']
        else :
            div_mmt[f'n={i}']=dict_df[im_id][column]-div_mmt[f'n={i-1}']
        mmt[im_id]=dict_df[im_id][column]
        i=i+1

    return mmt, div_mmt
