Paper: Deep Learning Approach to Improve Spatial Resolution of GOES-17 Wildfire Boundaries Using VIIRS Satellite Data
https://www.mdpi.com/2072-4292/16/4/715

This subfolder contains:

1: [CreateDatasetPipeline.py] User need to run this python file to create intermediate GOES and VIIRS tiff files and training dataset, this file contains following files to achieve it.
2: [CreateTrainingDataset.py] Finds contemporaneous files and use GOES and VIIRS processing to save them.
3: [GoesProcessing.py] Download, resample and reproject GOES files.
4: [VIIRSProcessing.py] Resample and reproject VIIRS files.
5: [WriteDataset4DLModel.py] writes dataset which is readable by Deep learning Models.
6: [SiteInfo.py] Helper class to have required information about the wildfire event such as longitude, latitude, start date and end date.
7: [GlobalValues.py] Helper file for global variables and constants.
8: [conf] Folder with wildfire event list and their properties to be used by create dataset process.


