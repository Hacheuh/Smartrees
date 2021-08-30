from PIL import Image
import glob
import ee
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# to make this code work, you have to create a folder named "output_gif" and another folder named "output_images". 
# in the first folder we are going to save the pngs images, in the second we will save  
# so in the terminal run the following lines:
# mkdir output_gif
# mkdir output_images

def output_images(df,name,shape):
    img_B10 = np.array(df['Norm_Temp'])
    img_NDVI = np.array(df['NDVI'])
    img_B10 = img_B10.reshape(shape) #  (377, 277) va sostituito con data.shapes['10'][:2]
    img_NDVI = img_NDVI.reshape(shape) 
    plt.figure(figsize=(15, 10))
    plt.imshow(img_B10, cmap='coolwarm')
    plt.savefig(f'output_images/{name}_Temp.png')
    plt.close()
    plt.figure(figsize=(15, 10))
    plt.imshow(img_NDVI, cmap='RdYlGn')
    plt.savefig(f'output_images/{name}_NDVI.png')
    plt.close()
    return None

def fill_value(data,size,value_ndvi,value_temp):
    df=pd.DataFrame(data.copy())
    true_size=len(df)
    df['indice']=df.index
    datframe=pd.DataFrame(np.zeros((size,))+value_ndvi,columns={'NDVI'})
    datframe['Norm_Temp']=np.zeros((size,))+value_temp
    datframe.loc[df.index,'NDVI']=df.loc[:,'NDVI']
    datframe.loc[df.index,'Norm_Temp']=df.loc[:,'Norm_Temp']
    return datframe

def create_gif_temp(pathname):
    '''This function transforms Temp pngs into a gif, and saves it    
    '''
    frames = []
    imgs = glob.glob(pathname+"/*_Temp.png")
    for i in imgs:
        new_frame = Image.open(i)
        frames.append(new_frame)
    print(frames)
    # Save into a GIF file that loops forever
    frames[0].save('output_gif/temp_gif.gif', format='GIF',
                    append_images=frames[1:],
                    save_all=True,
                    duration=650, loop=0)
    return None


def create_gif_NDVI(pathname):
    '''This function transforms pngs into a gif, and saves it    
    '''
    frames = []
    imgs = glob.glob(pathname+"/*_NDVI.png")
    for i in imgs:
        new_frame = Image.open(i)
        frames.append(new_frame)
    print(frames)        
    frames[0].save('output_gif/NDVI_gif.gif', format='GIF',
                    append_images=frames[1:],
                    save_all=True,
                    duration=650, loop=0)
    return None

def create_gifs_fromdf(datas_obj):
    dict_df=datas_obj.get_data_from_dates()
    shapes=datas_obj.shapes[10][:2]
    i=0
    for key in dict_df:
        i=i+1
        df=fill_value(dict_df[key],shapes[0]*shapes[1],-1, 5)
        output_images(df, f'Nice_{i}',shapes)
    create_gif_NDVI('output_images')
    create_gif_temp('output_images')
    return None
