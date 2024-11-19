"""
This Config file contains all the global constant used in Project

Created on Sun Jul 26 11:17:09 2020

@author:  mukul badhan
on Sun Jul 23 11:17:09 2022
"""
#  constants
FDC = "ABI-L2-FDCC"
RAD = "ABI-L1b-RadC"
CMI = "ABI-L2-CMIPC"
BETA = 'beta'
LC = 'LC'
HC = 'HC'
LI = 'LI'
HI = 'HI'
COLOR_NORMAL_VALUE = 255

GOES_MIN_VAL, GOES_MAX_VAL = [210] , [413]
VIIRS_MIN_VAL,VIIRS_MAX_VAL = 0 , 367
VIIRS_UNITS ='Brightness Temperature'

GOES_product = [{'product_name': RAD, 'band': 7}]
GOES_product_size = len(GOES_product)
GOES_Bands = 1
seperate_th,th_neg = 0,0

no_postfix = ''

#dataset creation postfix
site_Postfix = no_postfix
referenceDir_speficic_Postfix = no_postfix
write_trainingDir_speficic_Postfix = no_postfix

# model training and evaluation postfix 
trainingDir_speficic_Postfix = no_postfix
model_specific_postfix = no_postfix

# result changes postfix
result_specific_postfix = no_postfix

# real time model
realtime_model_specific_postfix = no_postfix


gf_c_fields = [f'gf_c{i+1}' for i in range(GOES_Bands)]
training_data_field_names = ['vf'] + gf_c_fields + ['vf_FRP', 'gf_min', 'gf_max', 'vf_max']

# GOES_UNITS = 'Radiance'
GOES_UNITS = 'Brightness Temperature'
PREDICTION_UNITS = 'Brightness Temperature'
RES_OPT_PTH = 'SuperRes_Opt.pth'
RES_DECODER_PTH = 'SuperRes_Decoder.pth'
RES_ENCODER_PTH = 'SuperRes_Encoder.pth'
RES_AUTOENCODER_PTH = 'SuperRes_AutoEncoder.pth'
LEARNING_RATE = 'learning_rate'
BATCH_SIZE = 'batch_size'
LOSS_FUNCTION = 'loss_function'
EPOCHS = 'epochs'
random_state = 42

# files and directories
GOES_ndf = 'GOES_netcdfs'
# GOES_ndf = '/home/mbadhan/WildFireGoes17/GOES_netcdfs'
goes_folder = "GOES"
viirs_folder = "VIIRS"
logs = 'logs'
compare = 'compare'

# data loading and preprocessing
site_conf = 'config/conf_sites.yml'
toExecuteSiteList = f"config/training_sites{site_Postfix}"
realtime_site_conf = 'config/conf_blind_testing.yml'

DataRepository = 'DataRepository'
reference_data = f"{DataRepository}/reference_data{referenceDir_speficic_Postfix}"
compare_dir = f'{reference_data}/compare/$LOC/'
# compare_dir = f'{reference_data}/compare_all/'
viirs_dir = f'{reference_data}/$LOC/VIIRS/'
goes_dir = f'{reference_data}/$LOC/GOES/$PROD_BAND/tif/'
goes_OrtRec_dir = f'{reference_data}/$LOC/GOES/$PROD_BAND/OrtRec_tif/'
write_training_dir = f'{DataRepository}/training_data{write_trainingDir_speficic_Postfix}/'
training_dir = f'{DataRepository}/training_data{trainingDir_speficic_Postfix}/'
# training_dir = 'training_data_working/'
GOES_OVERWRITE = False
VIIRS_OVERWRITE = False

GOES_tiff_file_name = 'GOES-{fire_date}_{ac_time}.tif'
GOES_OrtRec_tiff_file_name = 'GOES-{fire_date}_{ac_time}.tif'
VIIRS_tiff_file_name = 'viirs-snpp-{fire_date}_{ac_time}.tif'

# Autoencoder training and testing
# model_path = 'Model_BEFORE_MOVING_NORMALIZATION/'
model_path = 'Model/'
project_name_template = "{model_name}_{loss_function_name}_{n_epochs}epochs_{batch_size}batchsize_{learning_rate}lr{model_specific_postfix}"
test_split = 0.2
validation_split = 0.2
Results = f'{DataRepository}/results{result_specific_postfix}/'
# THRESHOLD_COVERAGE = 0.2
# THRESHOLD_IOU = 0.05
# THRESHOLD_COVERAGE, THRESHOLD_IOU = 0.2,0.05
THRESHOLD_COVERAGE, THRESHOLD_IOU = 0.453186035,0.005117899
# THRESHOLD_COVERAGE, THRESHOLD_IOU = 0.006,0.04


# toExecuteSiteList = "config/testing_sites"
testing_dir = f'{DataRepository}/testing_dir/'
# realtimeSiteList = "config/realtime_sites"
RealTimeIncoming_files = f'{DataRepository}/RealTimeIncoming_files/$LOC/$RESULT_TYPE/'
RealTimeIncoming_results = f'{DataRepository}/RealTimeIncoming_results/$LOC/$RESULT_TYPE/'
videos = f'{DataRepository}/Videos/'

# blind testing
realtimeSiteList = "config/blind_testing_sites"

paper_results = ['713','122','956','728','118','553','408','387','849','104','663','609']
NO_SAMPLES = []
RANDOM_SAMPLES = [str(i) for i in range(7000) if i % 400 == 0]
ALL_SAMPLES = 0
SELECTED_SAMPLES = NO_SAMPLES

image_source_path_for_Web = 'DataRepository/reference_data_b4_2022_west/compare'
