import streamlit as st
from PIL import Image
from streamlit_folium import folium_static
import ee
from datetime import timedelta
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import base64


from google.auth import compute_engine


import os
import json
import ee

# tout ça pour ça
json_str = os.getenv('GOOGLE_APPLICATION_CREDENTIALS42')

json_data = json.loads(json_str)
json_data['private_key'] = json_data['private_key'].replace('\\n', '\n')

outfile=open('test.json','w')
json.dump(json_data,outfile,indent=11)
outfile.close()

service_account = '9319975735-compute@developer.gserviceaccount.com '
credentials = ee.ServiceAccountCredentials(service_account,'test.json')
ee.Initialize(credentials)
os.remove('test.json')

"""ee_image='LANDSAT/LC08/C01/T1_TOA/LC08_195030_20210729'
img = ee.Image(ee_image).select(['B10'])

credentials = compute_engine.Credentials(scopes=['https://www.googleapis.com/auth/earthengine'])
ee.Initialize(credentials)"""



# alternative icon image
icon = Image.open("api/imgs/SmarTree.JPG")

st.set_page_config(layout='centered', page_title='SmarTrees', page_icon=icon)

def _max_width_(prcnt_width:int = 25):
    max_width_str = f"max-width: {prcnt_width}%;"
    st.markdown(f"""
                <style>
                .reportview-container .main .block-container{{{max_width_str}}}
                </style>
                """,
                unsafe_allow_html=True,
    )


CSS = """
h1 {
    color: DarkGreen;
}
h2 {
    color: DarkGreen;
}
h3 {
    color: DarkGreen;
}

.stApp {
    background-image: url(https://wallpaperaccess.com/full/1429640.jpg);
    background-size: cover;
}
"""
st.write(f'<style>{CSS}</style>', unsafe_allow_html=True)


st.markdown("""
    # SmarTrees
    ## 'Let trees cool down your city!'
    ###
""")

if st.button('Authentify'):
    ee.Authenticate()
    #st.echo(ee.Initialize())

st.markdown("""

    #### Choose a city and a date, discover in which areas trees are needed the most
    ####
""")

# input city name
pos = st.text_input('Select city coordinates', '7.25, 43.7')

col1, col2 = st.columns(2)


# input start date
date0 = col1.text_input('Beginning date', '2019-07-01')

#input end date
date1 = col2.text_input('End date', '2019-08-31')

st.markdown("""
    ## Interactive map of temperature and vegetation index
""")

import Smartrees.get_dataFrame as smgdf
import Smartrees.ee_query as smeq

Pos = pos.split(',')
pos_c=[float(Pos[0]),float(Pos[1])]
close_date = smeq.closest_image(date0, formatDate = 1, pos= tuple([float(item) for item in Pos]))

#st.write(close_date+timedelta(days=1))


# affichage gif dans le site
import Smartrees.pngs_to_gif as smptg
import Smartrees.date_to_data as smdtd
import os



df = smeq.get_meta_data(date_start  = str(close_date) , date_stop = str(close_date+timedelta(days=1)), pos = tuple([float(item) for item in Pos]))
st.write('Selected date is ',close_date)
smart=smgdf.SmarTrees(ee_image=df.loc[0,'id'], pos = tuple([float(item) for item in Pos]))
df1=smart.get_NDVIandKELVIN()
minNDVI=df1['NDVI'].min()
maxNDVI=df1['NDVI'].max()
#mintemp=df1['B10'].min()-273.15
#maxtemp=df1['B10'].max()-273.15
m = smart.display_folium_map(min_temp=0, max_temp=40, minNDVI=minNDVI, maxNDVI=maxNDVI)
folium_static(m)




st.markdown("""
    ## Areas of interest
    #### Here is the map highlighting the areas that can be improved
    ###
""")

col1,col2,col3=st.columns([1,3,1])
if st.button('Show'):
    df,shapes = smart.z_temperature()
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
    temp=df['Norm_Temp']
    ndvi=df['NDVI']
    index=temp.map(custom_index_temp)*ndvi.map(custom_index_ndvi)
    Tree_necessity_index=(index+abs(min(index)))/(max(index)-min(index))*100

    def fill_zeros(data=Tree_necessity_index,size=shapes[10][0]*shapes[10][1]):
        df=pd.DataFrame(data.copy())
        true_size=len(df)
        df['indice']=df.index
        datframe=pd.DataFrame(np.zeros((size,)),columns={'Tree_index'})
        datframe.loc[df.index,'Tree_index']=df.iloc[:,0]
        return datframe

    Tree_necessity_index_filled =fill_zeros()

    fig,ax=plt.subplots(figsize=(5,5))
    im=np.array(Tree_necessity_index_filled).reshape(shapes[10])
    s = ax.imshow(im, cmap='viridis')
    ax.set_title('Tree necessity index')
    ax.set_xticks([])
    ax.set_yticks([])
    fig.colorbar(s)
    col2.pyplot(fig)

st.markdown("""
    ## GIF viewer
""")

if st.button('Display gif'):


    file_ = open("api/imgs/NDVI_ok.gif", "rb")
    contents = file_.read()
    data_url = base64.b64encode(contents).decode("utf-8")
    file_.close()

    col5,col6=st.columns([2,2])
    col5.markdown("""
    #### Gif of Vegetation Index
    #
    """)
    col5.markdown(
        f'<img src="data:image/gif;base64,{data_url}" alt="cat gif">',
        unsafe_allow_html=True,
    )

    file_ = open("api/imgs/temp_ok.gif", "rb")
    contents = file_.read()
    data2_url = base64.b64encode(contents).decode("utf-8")
    file_.close()

    col6.markdown("""
    #### Gif of temperature
    #
    """)
    col6.markdown(
        f'<img src="data:image/gif;base64,{data2_url}" alt="cat gif">',
        unsafe_allow_html=True,
    )
