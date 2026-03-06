## Dataset Preparation and Uploads

This project uses geospatial datasets extracted using Google Earth Engine.

Due to GitHub file size limitations, the dataset is not stored directly in the repository.

To generate the dataset:
1. Sign up for Google Earth Engine.
2. Authenticate Earth Engine in Python and connect your project.
3. Run the notebook:
  data_preperation/Hydrology_Engine_GIS_Data_Preparation.ipynb
4. To get delhi_wards.kml view this folder: 
  Frontend/assets/delhi_wards.kml

This will export the required GeoTIFF files to your drive and generate the training dataset automatically.

