import streamlit as st
from PIL import Image

# alternative icon image
icon = Image.open("raw_data/SmarTree.JPG")

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
city = st.text_input('select a city name', 'Nice')
st.write('You have selected', city)

# input start date
date0 = st.text_input('Beginning date', '')
st.write('You have selected', date0)

#input end date
date1 = st.text_input('End date', '')
st.write('You have selected', date1)

st.markdown("""
    ## Cartographic representation

    ### 1...
    ## load results maps n°1
""")



st.markdown("""
    ### 2...
    ## load results maps n°2
""")
