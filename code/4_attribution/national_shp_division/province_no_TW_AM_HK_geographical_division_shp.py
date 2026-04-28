import os
import glob
import geopandas as gpd

home_dir = os.path.expanduser("/path/to/home/")
base_path = os.path.join(home_dir, "PROJECT_DIR", "Papers", "4_NC")
province_no_TW_AM_HK_shp = os.path.join(
    base_path, "data", "Administrative_divisions_of_china", "no_TW_AM_HK", "china_pro2_no_TW_AM_HK.shp"
)
province_no_TW_AM_HK_geographical_division_shp = os.path.join(
    base_path, "data", "Administrative_divisions_of_china", "no_TW_AM_HK", "china_pro2_no_TW_AM_HK_geographical_division.shp"
)

def to_int_adcode(x):
    if x is None:
        return None
    try:
        return int(x)
    except Exception:
        try:
            return int(float(x))
        except Exception:
            return None

# ADCODE99 -> (DIV_CN, DIV_EN)
adcode_to_div = {
    # 华北
    110000: ("华北", "North China"),
    120000: ("华北", "North China"),
    130000: ("华北", "North China"),
    140000: ("华北", "North China"),
    150000: ("华北", "North China"),
    370000: ("华北", "North China"),
    # 东北
    210000: ("东北", "Northeast China"),
    220000: ("东北", "Northeast China"),
    230000: ("东北", "Northeast China"),
    # 华东
    310000: ("华东", "East China"),
    320000: ("华东", "East China"),
    330000: ("华东", "East China"),
    340000: ("华东", "East China"),
    # 华南
    350000: ("华南", "South China"),
    440000: ("华南", "South China"),
    450000: ("华南", "South China"),
    460000: ("华南", "South China"),
    # 华中
    410000: ("华中", "Central China"),
    420000: ("华中", "Central China"),
    430000: ("华中", "Central China"),
    360000: ("华中", "Central China"),
    # 西南
    500000: ("西南", "Southwest China"),
    510000: ("西南", "Southwest China"),
    520000: ("西南", "Southwest China"),
    530000: ("西南", "Southwest China"),
    540000: ("西南", "Southwest China"),
    # 西北
    610000: ("西北", "Northwest China"),
    620000: ("西北", "Northwest China"),
    630000: ("西北", "Northwest China"),
    640000: ("西北", "Northwest China"),
    650000: ("西北", "Northwest China"),
}

# ADCODE99 -> NAME_EN_JX（按你给的表）
adcode_to_name_en = {
    510000: "Sichuan",
    460000: "Hainan",
    410000: "Henan",
    310000: "Shanghai",
    120000: "Tianjin",
    110000: "Beijing",
    540000: "Tibet",
    630000: "Qinghai",
    640000: "Ningxia",
    350000: "Fujian",
    620000: "Gansu",
    530000: "Yunnan",
    320000: "Jiangsu",
    330000: "Zhejiang",
    230000: "Heilongjiang",
    220000: "Jilin",
    150000: "Inner Mongolia",
    610000: "Shaanxi",
    340000: "Anhui",
    500000: "Chongqing",
    650000: "Xinjiang",
    130000: "Hebei",
    420000: "Hubei",
    440000: "Guangdong",
    210000: "Liaoning",
    370000: "Shandong",
    520000: "Guizhou",
    360000: "Jiangxi",
    430000: "Hunan",
    450000: "Guangxi",
    140000: "Shanxi",
}

# 读取
gdf = gpd.read_file(province_no_TW_AM_HK_shp)
if "ADCODE99" not in gdf.columns:
    raise KeyError(f"输入 shp 缺少字段 ADCODE99。现有字段：{list(gdf.columns)}")

gdf["_ADCODE99_INT"] = gdf["ADCODE99"].apply(to_int_adcode)

# 生成分区列（Shapefile 字段名<=10，所以用 DIV_CN / DIV_EN）
gdf["DIV_CN"] = gdf["_ADCODE99_INT"].map(lambda k: adcode_to_div.get(k, (None, None))[0])
gdf["DIV_EN"] = gdf["_ADCODE99_INT"].map(lambda k: adcode_to_div.get(k, (None, None))[1])

# 确保 NAME_EN_JX 存在并补齐
if "NAME_EN_JX" not in gdf.columns:
    gdf["NAME_EN_JX"] = None
gdf["NAME_EN_JX"] = gdf["NAME_EN_JX"].fillna(gdf["_ADCODE99_INT"].map(adcode_to_name_en))

# 检查未匹配
unmatched = gdf[gdf["DIV_CN"].isna() | gdf["DIV_EN"].isna() | gdf["NAME_EN_JX"].isna()]
if len(unmatched) > 0:
    bad = sorted(set(unmatched["_ADCODE99_INT"].tolist()))
    raise ValueError(f"存在未匹配/缺失的 ADCODE99：{bad}。请检查输入 shp 是否包含异常记录。")

# 只保留你要的列（含 NAME_EN_JX）
gdf_out = gdf[["ADCODE99", "NAME_EN_JX", "DIV_CN", "DIV_EN", "geometry"]].copy()

# 清理旧输出同名文件组
out_dir = os.path.dirname(province_no_TW_AM_HK_geographical_division_shp)
os.makedirs(out_dir, exist_ok=True)
base = os.path.splitext(province_no_TW_AM_HK_geographical_division_shp)[0]
for f in glob.glob(base + ".*"):
    try:
        os.remove(f)
    except Exception:
        pass

# 保存
gdf_out.to_file(
    province_no_TW_AM_HK_geographical_division_shp,
    driver="ESRI Shapefile",
    encoding="UTF-8",  # 如 ArcGIS 中文乱码可改 GBK
)

print("✅ 输出完成：", province_no_TW_AM_HK_geographical_division_shp)
print("输出字段：ADCODE99, NAME_EN_JX, DIV_CN, DIV_EN, geometry")
