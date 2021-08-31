import streamlit as st
from PIL import Image
from streamlit_folium import folium_static
import ee
from datetime import timedelta

# alternative icon image
icon = Image.open("api/imgs/SmarTree.JPG")

st.set_page_config(layout='centered', page_title='SmarTrees', page_icon=icon)

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

    ## Get predictions on where you need to plant trees!

    ### Predict average temperature based on potential Vegetation Index (NDVI) that could be obtained by planting trees 
""")

# input city name
pos = st.text_input('select a city name', '7.25, 43.7')
st.write('You have selected', pos)

# input start date
date0 = st.text_input('Beginning date', '2017-01-31')
st.write('You have selected', date0)

#input end date
date1 = st.text_input('End date', '')
st.write('You have selected', date1)

st.markdown("""
    ## Cartographic representation

    ### 1...
    ## load results maps n°1
""")

import Smartrees.get_dataFrame as smgdf
import Smartrees.ee_query as smeq
ee.Initialize()
Pos = pos.split(',')
close_date = smeq.closest_image(date0, formatDate = 1, pos= tuple([float(item) for item in Pos]))

#st.write(close_date+timedelta(days=1))

df = smeq.get_meta_data(date_start  = str(close_date) , date_stop = str(close_date+timedelta(days=1)), pos = tuple([float(item) for item in Pos]))
st.write('selected date is ',close_date)
smart=smgdf.SmarTrees(ee_image=df.loc[0,'id'], pos = tuple([float(item) for item in Pos]))
df1=smart.get_NDVIandKELVIN()
minNDVI=df1['NDVI'].min()
maxNDVI=df1['NDVI'].max()
#mintemp=df1['B10'].min()-273.15
#maxtemp=df1['B10'].max()-273.15
m = smart.display_folium_map(min_temp=0, max_temp=40, minNDVI=minNDVI, maxNDVI=maxNDVI)
folium_static(m)




st.markdown("""
    ### 2...
    ## load results maps n°2
""")
