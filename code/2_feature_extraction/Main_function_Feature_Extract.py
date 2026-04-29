import os
import pandas as pd
import os
import geopandas as gpd
from shapely.geometry import Point
# import rasterio
import numpy as np
# from rasterio.transform import from_origin
from tqdm import tqdm  # is used to display the progress bar
###########################################################################################################
# ##########################################Select path################################################
###########################################################################################################
###########################################################################################################
database_folder_path = os.path.join("/path/to/home/", "Large_scale", "future_dataset")
input_folder_path = os.path.join("/path/to/home/", "PROJECT_DIR", "Papers", "4_NC", "data")
# ###output_folder_path, around line 105
print("\ninput_folder_path:\n", input_folder_path)
###########################################################################################################
# ##############################Manually modified variables########################################################
###########################################################################################################
###########################################################################################################
df_path = os.path.join(
    input_folder_path, "points",                   # Nationwide, 64 lines of sinkhole_position need to be changed simultaneously

# =============== National positive and negative sample training data extraction ===============
    # "Positive_Negative_balanced_25366.csv" # Positive sample + balanced negative sample
    "Points_China_all_10km.csv"                          # 10km
)
###########################################################################################################
# ##################################18 Groundwater does not have ssp2 and ssp4; 17, 20, 21, 22, 23 does not have ssp4; ##########################
###########################################################################################################
ssp = "ssp4"  # TODO: Manually change to ssp1/ ssp2/ssp3/ssp4/ssp5/ as needed. Select 'ssp2' to comment out 18 groundwater, open the requirement 10.6 / Select 'ssp4' to comment out 10 leaf area index, 17 rainfall, 18 groundwater, 20, 21, 22, 23, open the requirement 10.7
###########################################################################################################
###########################################################################################################
###########################################################################################################
###########################################################################################################
df = pd.read_csv(df_path)

# Extract key columns
# sinkhole_position = df[['No', 'Disaster', 'Longitude', 'Latitude', 'ADCODE99', 'NAME_EN_JX']].copy() # nationwide sinkhole
sinkhole_position = df[['No', 'Disaster', 'Longitude', 'Latitude', 'ADCODE99', 'NAME_EN_JX']].copy()  # Processing step.

# Check results
print(sinkhole_position.head())
print(f"\n extraction completed! total{len(sinkhole_position)}records")
###########################################################################################################
# #####################################Create a mapping table from input to output paths#########################################
###########################################################################################################
###########################################################################################################
# ========= : df_path output_folder_path =========
# File name -> subfolder name mapping
csv_to_subfolder = {
# =============== National positive and negative sample training data extraction ===============
    "Positive_Negative_balanced_25366.csv": "Positive_Negative_balanced",
    "Points_China_all_10km.csv": "Points_China_all_10km",

}
# Get the file name from df_path
csv_name = os.path.basename(df_path)
if csv_name not in csv_to_subfolder:
    raise ValueError(f"Unknown df_path filename:{csv_name}, please add mapping in csv_to_subfolder.")
output_folder_path = os.path.join(input_folder_path, "Extracted_HAVE_future",csv_to_subfolder[csv_name])
# If necessary, you can make sure the directory exists
os.makedirs(output_folder_path, exist_ok=True)
print("\\n Save the extracted features to the folder: \\n", output_folder_path)
###########################################################################################################
###########################################################################################################
###########################################################################################################
###########################################################################################################
# Create separate subfolders for historical/future data, and create SSP subfolders for future scenarios
###########################################################################################################

# 1) Create two new subfolders, historical and future, under output_folder_path
historical_folder_path = os.path.join(output_folder_path, "historical")
future_folder_path = os.path.join(output_folder_path, "future")

os.makedirs(historical_folder_path, exist_ok=True)
os.makedirs(future_folder_path, exist_ok=True)

# 3) Under the future folder, create a subfolder for the current SSP
future_ssp_folder_path = os.path.join(future_folder_path, ssp)
os.makedirs(future_ssp_folder_path, exist_ok=True)

print("\\n historical data output folder: \\n", historical_folder_path)
print("( SSP ):\\n", future_ssp_folder_path)
###########################################################################################################

###########################################################################################################
# ################# Calculate the closest distance to the karst area and save it to historical_folder_path/Distance_to_karst.csv##############
###########################################################################################################
from distance_to_karst import distance_to_karst
sinkhole_position = distance_to_karst(
    sinkhole_position,
    database_folder_path,
    historical_folder_path,
)
###########################################################################################################
# ############### Extract bedrock depth information and save it to historical_folder_path/Depth_to_Bedrock.csv####################
###########################################################################################################
from depth_to_bedrock import depth_to_bedrock
sinkhole_position = depth_to_bedrock(
    sinkhole_position,
    database_folder_path,
    historical_folder_path,
)
###########################################################################################################
# #################### Calculate the closest distance to the fault and save it to historical_folder_path/Distance_to_Fault.csv#############
###########################################################################################################
from distance_to_fault import distance_to_fault
sinkhole_position = distance_to_fault(
    sinkhole_position,
    database_folder_path,
    historical_folder_path,
)
##########################################################################################################
# ########### ,historical_folder_pathfuture_ssp_folder_path#########
##########################################################################################################
from urban_land_fraction import urban_land_fraction
sinkhole_position = urban_land_fraction(
    sinkhole_position,
    database_folder_path,
    historical_folder_path,
    future_ssp_folder_path,
    ssp,
)
###########################################################################################################
# ######################## Calculate the total population and save it to historical_folder_path and future_ssp_folder_path##############
###########################################################################################################
from population_total import population_total
sinkhole_position = population_total(
    sinkhole_position,
    database_folder_path,
    historical_folder_path,
    future_ssp_folder_path,
    ssp,
)
###########################################################################################################
# ######## Impermeability index (0-100, 0 is water permeable, 100 is impermeable), and saved to historical_folder_path and future_ssp_folder_path#####
###########################################################################################################
from impervious_index import impervious_index
sinkhole_position = impervious_index(
    sinkhole_position,
    database_folder_path,
    historical_folder_path,
    future_ssp_folder_path,
    ssp,
)
# ###########################################################################################################
# ######### Leaf area index (LAI) (temporal resolution is monthly, spatial resolution is 2.5 arcs), and save ###################################
# ###########################################################################################################
# from leaf_area_index import leaf_area_index
# sinkhole_position = leaf_area_index(
#     sinkhole_position,
#     database_folder_path,
#     historical_folder_path,
#     future_ssp_folder_path,
#     ssp,
# )
# ##########################################################################################################
# ####################### Precipitation, resolution 0.25 degrees (25 km), time resolution annual ###################################
# ##########################################################################################################
# from precipitation_amount import precipitation_amount
# sinkhole_position = precipitation_amount(
#     sinkhole_position,
#     database_folder_path,
#     historical_folder_path,
#     future_ssp_folder_path,
#     ssp,
# )

# # ###########################################################################################################
# # ################################################ 20_tas###################################################
# # ###########################################################################################################
# from tas import tas
# sinkhole_position = tas(
#     sinkhole_position,
#     database_folder_path,
#     historical_folder_path,
#     future_ssp_folder_path,
#     ssp,
# )

# # ###########################################################################################################
# # ################################################ 21_tasmax################################################
# # ###########################################################################################################
# from tasmax import tasmax
# sinkhole_position = tasmax(
#     sinkhole_position,
#     database_folder_path,
#     historical_folder_path,
#     future_ssp_folder_path,
#     ssp,
# )

# # ###########################################################################################################
# # ################################################ 22_tasmin################################################
# # ###########################################################################################################
# from tasmin import tasmin
# sinkhole_position = tasmin(
#     sinkhole_position,
#     database_folder_path,
#     historical_folder_path,
#     future_ssp_folder_path,
#     ssp,
# )

# # ###########################################################################################################
# # ################################################ 23_huss##################################################
# # ###########################################################################################################
# from huss import huss
# sinkhole_position = huss(
#     sinkhole_position,
#     database_folder_path,
#     historical_folder_path,
#     future_ssp_folder_path,
#     ssp,
# )

# #########################################################################################################
# ####### Groundwater: factors wtd - Water Table Depth
# #########################################################################################################
# from groundwater_wtd import groundwater_wtd
# sinkhole_position = groundwater_wtd(
#     sinkhole_position,
#     database_folder_path,
#     historical_folder_path,
#     future_ssp_folder_path,
#     ssp,
# )
# ##########################################################################################################
# ######### Groundwater: Factors hds - Groundwater Head
# ###########################################################################################################
# from groundwater_hds import groundwater_hds
# sinkhole_position = groundwater_hds(
#     sinkhole_position,
#     database_folder_path,
#     historical_folder_path,
#     future_ssp_folder_path,
#     ssp,
# )
# ###########################################################################################################
# ########################################10.51:##############################################
# ###########################################################################################################
from feature_state_io import save_feature_state
save_feature_state(
    df_path=df_path,
    ssp=ssp,
    sinkhole_position=sinkhole_position,
    database_folder_path=database_folder_path,
    input_folder_path=input_folder_path,
    output_folder_path=output_folder_path,
    historical_folder_path=historical_folder_path,
    future_folder_path=future_folder_path,
    future_ssp_folder_path=future_ssp_folder_path,
    # If you have other variables you want to save, such as csv_to_subfolder, you can also:
    # extra_vars={"csv_to_subfolder": csv_to_subfolder}
)
# ###########################################################################################################
# ######################################Requirement 10.52: Global variable call ###########################################
# ###########################################################################################################
# from feature_state_io import load_feature_state

# state = load_feature_state(df_path, ssp)

# sinkhole_position = state["sinkhole_position"]
# database_folder_path = state["database_folder_path"]
# input_folder_path = state["input_folder_path"]
# output_folder_path = state["output_folder_path"]
# historical_folder_path = state["historical_folder_path"]
# future_folder_path = state["future_folder_path"]
# future_ssp_folder_path = state["future_ssp_folder_path"]

# ###########################################################################################################
# #####################################Requirement 10.6: Interpolate the missing 10 columns of data in ssp2##################################
# ###########################################################################################################
# from post_processing_for_ssp2 import post_processing_for_ssp2

# sinkhole_position = post_processing_for_ssp2(
#     sinkhole_position=sinkhole_position,
#     df_path=df_path,
#     historical_folder_path=historical_folder_path,
#     future_folder_path=future_folder_path,
#     future_ssp_folder_path=future_ssp_folder_path,
#     ssp=ssp,
# )
###########################################################################################################
# ####################################Requirement 10.7: Interpolate the 20 missing columns of data in ssp4##################################
###########################################################################################################
from post_processing_for_ssp4 import post_processing_for_ssp4

sinkhole_position = post_processing_for_ssp4(
    sinkhole_position=sinkhole_position,
    df_path=df_path,
    historical_folder_path=historical_folder_path,
    future_folder_path=future_folder_path,
    future_ssp_folder_path=future_ssp_folder_path,
    ssp=ssp,
)
###########################################################################################################
# #################################11:sinkhole_positioncsv################################
###########################################################################################################
###########################################################################################################
from aggregate_features import aggregate_features
sinkhole_position = aggregate_features(
    sinkhole_position=sinkhole_position,
    df_path=df_path,
    output_folder_path=output_folder_path,  # corresponds to the directory mapped by csv_to_subfolder
    ssp=ssp,  # SSP,; None
)
###########################################################################################################
# #########Requirement 12: Clean + scramble + minimize the summarized sinkhole_position to prevent singular solutions, and output *_cleaned.csv#############
###########################################################################################################
from data_clean_shuffle import data_clean_and_shuffle
sinkhole_position_cleaned = data_clean_and_shuffle(
    sinkhole_position=sinkhole_position,
    df_path=df_path,
    output_folder_path=output_folder_path,  # corresponds to the directory mapped by csv_to_subfolder
    ssp=ssp,  # SSP,; None
    random_state=42,
)
##########################################################################################################