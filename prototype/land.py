import math
import sklearn
import numpy as np

class Land:
    def __init__(self, zoning_description = 0, zoning_type = 0, zoning_subtype = 0, lbcs_structure_desc = 0,
                 lat = 0, lon = 0, ll_gissqft = 0, county = 0):
        self.lbcs_structure_desc = self.quant_zoning(lbcs_structure_desc)

        self.county = float(self.quant_county(county))
        self.lat = float(lat)
        self.lon = float(lon)
        self.ll_gissqft = float(ll_gissqft)
        self.haversine = float(self.add_haversine())

    def __str__(self):
        return f"{self.county}"
    
    def to_array(self):
        return [[self.haversine, self.county, self.ll_gissqft], [self.lbcs_structure_desc]]

    def add_haversine(self, r = 6371000):
        #print(other)
        lat1, lat2 = math.radians(self.lat), math.radians(42.3394)
        long1, long2 = math.radians(self.lon), math.radians(-71.0940)

        lat_delta = lat2 - lat1
        long_delta = long2 - long1

        a = (math.sin(lat_delta / 2)) ** 2 + math.cos(lat1) * math.cos(lat2) * \
        (math.sin(long_delta / 2)) ** 2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

        return c * 6371000
   
    def quant_county(self, county):
        COUNTIES = ["barnstable", "berkshire", "bristol", "dukes", "essex", \
            "franklin", "hampden", "hampshire", "middlesex", "nantucket", \
            "norfolk", "plymouth", "suffolk", "worcester"]
        if county in COUNTIES:
            return COUNTIES.index(county)
        else: 
            return 0
    def quant_zoning(self, desc):
        zoning_map = DESCRIPTIONS = ['Industrial buildings and structures', 'Single-family buildings', \
                               'Multifamily structures: Two Units', 'Residential buildings', \
                                'Electric lines, phone and cable lines, etc.', 'Electric substation and distribution facility', \
                                'Churches, synagogues, temples, mosques, etc.', 'Store or shop building', 'Multifamily structures', \
                                'Commercial buildings and other specialized structures', 'Multifamily structures: Three Units', \
                                'Office or store building with residence on top', 'Office or bank building', 'School or university buildings', \
                                'Cemetery, monument, tombstone, or mausoleum']
        if desc in DESCRIPTIONS:
            return DESCRIPTIONS.index(desc)
        else:
            return 0