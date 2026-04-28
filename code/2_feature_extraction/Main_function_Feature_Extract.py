import os
import pandas as pd
import os
import geopandas as gpd
from shapely.geometry import Point
# import rasterio
import numpy as np
# from rasterio.transform import from_origin
from tqdm import tqdm  # 用于显示进度条
###########################################################################################################
##############################################选择路径#####################################################
###########################################################################################################
###########################################################################################################
database_folder_path = os.path.join("/path/to/home/", "Large_scale", "future_dataset")
input_folder_path = os.path.join("/path/to/home/", "PROJECT_DIR", "Papers", "4_NC", "data")
####output_folder_path，在105行左右
print("\ninput_folder_path:\n", input_folder_path)
###########################################################################################################
#################################手动修改的变量#############################################################
###########################################################################################################
###########################################################################################################
df_path = os.path.join(
    input_folder_path, "points",                   # 全国       需要同步更改64行sinkhole_position

# =============== 全国范围的正负样本训练数据提取 ===============
    # "Positive_Negative_balanced_25366.csv"               # 正样本+平衡负样本
    "Points_China_all_10km.csv"                          # 全国范围无正样本的均匀网格数据10km分辨率
)
###########################################################################################################
#####################################18地下水没有ssp2和ssp4；17,20,21,22,23没有ssp4；###########################
###########################################################################################################
ssp = "ssp4"  # TODO: 根据需要手动改成 ssp1/ ssp2/ssp3/ssp4/ssp5/ 选择'ssp2'注释掉18地下水，打开需求10.6 /选择 'ssp4'注释掉10叶面积指数，17降雨，18地下水,20,21,22,23,打开需求10.7
###########################################################################################################
###########################################################################################################
###########################################################################################################
###########################################################################################################
df = pd.read_csv(df_path)

# 提取关键列
# sinkhole_position = df[['No', 'Disaster', 'Longitude', 'Latitude', 'ADCODE99', 'NAME_EN_JX']].copy()  #全国范围有地陷
sinkhole_position = df[['No', 'Disaster', 'Longitude', 'Latitude', 'ADCODE99', 'NAME_EN_JX']].copy()  #全国范围无地陷

# 检查结果
print(sinkhole_position.head())
print(f"\n提取完成！共 {len(sinkhole_position)} 条记录")
###########################################################################################################
########################################创建输入到输出路径的映射表###########################################
###########################################################################################################
###########################################################################################################
# ========= 关键判断部分：根据 df_path 选择 output_folder_path =========
# 文件名 -> 子文件夹名 的映射
csv_to_subfolder = {
# =============== 全国范围的正负样本训练数据提取 ===============
    "Positive_Negative_balanced_25366.csv": "Positive_Negative_balanced",
    "Points_China_all_10km.csv": "Points_China_all_10km",

}
# 从 df_path 里取出文件名
csv_name = os.path.basename(df_path)
if csv_name not in csv_to_subfolder:
    raise ValueError(f"未知的 df_path 文件名: {csv_name}，请在 csv_to_subfolder 中添加映射。")
output_folder_path = os.path.join(input_folder_path, "Extracted_HAVE_future",csv_to_subfolder[csv_name])
# 如果需要，可以顺便确保目录存在
os.makedirs(output_folder_path, exist_ok=True)
print("\n保存提取的特征到文件夹:\n", output_folder_path)
###########################################################################################################
###########################################################################################################
###########################################################################################################
###########################################################################################################
# 为历史 / 未来数据分别建立子文件夹，并为未来情景建立 SSP 子文件夹
###########################################################################################################

# 1) 在 output_folder_path 下新建 historical 和 future 两个子文件夹
historical_folder_path = os.path.join(output_folder_path, "historical")
future_folder_path = os.path.join(output_folder_path, "future")

os.makedirs(historical_folder_path, exist_ok=True)
os.makedirs(future_folder_path, exist_ok=True)

# 3) 在 future 文件夹下面，为当前 SSP 建立子文件夹
future_ssp_folder_path = os.path.join(future_folder_path, ssp)
os.makedirs(future_ssp_folder_path, exist_ok=True)

print("\n历史数据输出文件夹:\n", historical_folder_path)
print("未来数据输出文件夹(当前 SSP 情景):\n", future_ssp_folder_path)
###########################################################################################################

###########################################################################################################
################## 计算与岩溶区的最近距离，并保存到 historical_folder_path/Distance_to_karst.csv##############
###########################################################################################################
from distance_to_karst import distance_to_karst
sinkhole_position = distance_to_karst(
    sinkhole_position,
    database_folder_path,
    historical_folder_path,
)
###########################################################################################################
################# 提取基岩深度信息，并保存到 historical_folder_path/Depth_to_Bedrock.csv#####################
###########################################################################################################
from depth_to_bedrock import depth_to_bedrock
sinkhole_position = depth_to_bedrock(
    sinkhole_position,
    database_folder_path,
    historical_folder_path,
)
###########################################################################################################
##################### 计算与断层的最近距离，并保存到 historical_folder_path/Distance_to_Fault.csv#############
###########################################################################################################
from distance_to_fault import distance_to_fault
sinkhole_position = distance_to_fault(
    sinkhole_position,
    database_folder_path,
    historical_folder_path,
)
##########################################################################################################
############ 计算城市土地与所有土地面积的比例，并保存到historical_folder_path和future_ssp_folder_path#########
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
######################### 计算人口总数，并保存到historical_folder_path和future_ssp_folder_path###############
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
######### 不透水指数（0-100,0是透水，100是不透水），并保存到historical_folder_path和future_ssp_folder_path#####
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
# ######### 叶面积指数（LAI）（时间分辨率是按月，空间分辨率是2.5 弧），并保存#####################################
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
# ####################### 降水量，分辨率0.25度（25公里），时间分辨率为年度#####################################
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
# ####### 地下水：因素wtd - Water Table Depth
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
# ######### 地下水：因素hds - Groundwater Head
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
# ########################################需求10.51：全局变量暂存##############################################
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
    # 如果你还有其他变量想存，比如 csv_to_subfolder，也可以：
    # extra_vars={"csv_to_subfolder": csv_to_subfolder}
)
# ###########################################################################################################
# ########################################需求10.52：全局变量调用##############################################
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
# ########################################需求10.6：对ssp2缺失的10列数据插值###################################
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
########################################需求10.7：对ssp4缺失的20列数据插值###################################
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
##################################需求11：保存总的sinkhole_position到csv文件################################
###########################################################################################################
###########################################################################################################
from aggregate_features import aggregate_features
sinkhole_position = aggregate_features(
    sinkhole_position=sinkhole_position,
    df_path=df_path,
    output_folder_path=output_folder_path,  # 对应 csv_to_subfolder 映射后的目录
    ssp=ssp,  # 如果你希望区分不同 SSP，就传入；不想区分也可以写 None
)
###########################################################################################################
##########需求12：对汇总后的sinkhole_position做清洗 + 打乱+极小值防止奇异解，并输出 *_cleaned.csv#############
###########################################################################################################
from data_clean_shuffle import data_clean_and_shuffle
sinkhole_position_cleaned = data_clean_and_shuffle(
    sinkhole_position=sinkhole_position,
    df_path=df_path,
    output_folder_path=output_folder_path,  # 对应 csv_to_subfolder 映射后的目录
    ssp=ssp,  # 如果你希望区分不同 SSP，就传入；不想区分也可以写 None
    random_state=42,
)
##########################################################################################################