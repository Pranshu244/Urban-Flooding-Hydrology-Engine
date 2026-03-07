## Dataset Preparation and Uploads

This project uses geospatial datasets generated using Google Earth Engine.
Due to GitHub file size limitations, the datasets are not stored directly in this repository.

### Steps to Generate the Dataset

1. Create an account on Google Earth Engine.
2. Authenticate Earth Engine in Python and connect your project.
3. Run the notebook:

   data_preparation/Hydrology_Engine_GIS_Data_Preparation.ipynb

4. For the Delhi wards boundary file, refer to:

   Frontend/assets/delhi_wards.kml
   
Running the notebook will automatically export the required GeoTIFF files to your drive and generate the dataset used for model training.
