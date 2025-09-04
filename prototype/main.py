import pandas as pd
import csv
from land import Land
import os
from sklearn.impute import SimpleImputer
from predictor import *



FILENAME = "data/final_merged_ma.csv"
ATTRIBS = ["zoning_description", "zoning_type", "zoning_subtype", "lbcs_structure_desc", \
           "lat", "lon", "ll_gissqft", "county"]
CITYCENTER = Land(lat = 42.3394, lon = -71.0940)
COUNTIES = ["barnstable", "berkshire", "bristol", "dukes", "essex", \
            "franklin", "hampden", "hampshire", "middlesex", "nantucket", \
            "norfolk", "plymouth", "suffolk", "worcester"]
DESCRIPTIONS = ['Industrial buildings and structures', 'Single-family buildings', \
                               'Multifamily structures: Two Units', 'Residential buildings', \
                                'Electric lines, phone and cable lines, etc.', 'Electric substation and distribution facility', \
                                'Churches, synagogues, temples, mosques, etc.', 'Store or shop building', 'Multifamily structures', \
                                    'Commercial buildings and other specialized structures', 'Multifamily structures: Three Units', \
                                        'Office or store building with residence on top', 'Office or bank building', 'School or university buildings', \
                                            'Cemetery, monument, tombstone, or mausoleum']

def clean_data(data, keys):
    clean_lst = data.filter(items=ATTRIBS)
    imputer = SimpleImputer(strategy='most_frequent')
    imputed = pd.DataFrame(imputer.fit_transform(clean_lst), columns=clean_lst.columns)
    return imputed

def create_lands(data):
    lands = []
    for index, row in data.iterrows():
        land = Land(**row.to_dict())
        land.add_haversine(CITYCENTER)
        lands.append(land)
    return lands

def main():
    data = pd.read_csv(FILENAME)
    data = clean_data(data, ATTRIBS)
    lands = create_lands(data)
    clean_lands = []
    for land in lands:
        if not land.lbcs_structure_desc == "":
            clean_lands.append(land)
            
    ai = Predictor(clean_lands)
    ai.displayPrediction()
    
main()
