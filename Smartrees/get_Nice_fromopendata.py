import pandas as pd
import folium
import ee
import geemap
import datetime
from IPython.display import Image, display
import geehydro
import os
def get_trees_coordinates(date=2019):
    csvpath = f"ev-arbre-opendata-{date}.csv"
    p = os.path.join("Smartrees/raw_data",csvpath)
    trees = pd.read_csv(p,sep = ';')
    all_coords = trees.geometry
    all_coords2 = all_coords.map(lambda x : eval(x))
    coordinates = [coords['coordinates'] for coords in all_coords2]
    return coordinates

def tree_maps(location = [43.6961,7.27178],zoom=10,date=2019):
    # Create a Map of Nice with trees only
    coords = get_trees_coordinates(date = date)
    Nice = folium.Map(location = location,zoom_start=zoom)
    trees = ee.Geometry.MultiPoint(coords)
    Nice.addLayer(ee.Image().paint(trees,"green",1),{},'Trees')
    Nice.setCenter(7.2717,43.696,zoom=13)
    return Nice

def get_vege_coordinates(date = 2020):
    vege2020 = pd.read_csv(f"Smartrees/raw_data/ev-inventaire-opendata-{date}.csv",sep = ';')
    vege2020 = vege2020.geometry.map(lambda x : eval(x))
    multivege20 = [coords['coordinates'] for coords in vege2020 if coords['type'] == 'MultiPolygon']
    vege20 = [coords['coordinates'] for coords in vege2020 if coords['type'] != 'MultiPolygon']
    return {"simplepoly" : vege20,"multipoly" : multivege20}

def vege_and_tree():
    Nice = tree_maps(location = [43.6961,7.27178],zoom=10,date=2020,set_options = 'SATELLITE')
    print(Nice)
    data = get_vege_coordinates()
    multivege = data["multipoly"]
    simplevege = data["simplepoly"]
    for i in range(len(multivege)):
        multivege201 = ee.Geometry.MultiPolygon(multivege[i])
        Nice.addLayer(ee.Image().paint(multivege201,1),{"color" : "662A00"},'colored')
    Nice.addLayer(ee.Image().paint(simplevege,10),{},'Vegetations')
    return Nice

def show_trees_map(date=2019):
    Nice = tree_maps(date=date)
    display(Nice)


def show_vege_and_trees_map():
    Nice = vege_and_tree()
    display(Nice)
