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
    # North China
    110000: ("North China", "North China"),
    120000: ("North China", "North China"),
    130000: ("North China", "North China"),
    140000: ("North China", "North China"),
    150000: ("North China", "North China"),
    370000: ("North China", "North China"),
    # Northeast
    210000: ("Northeast", "Northeast China"),
    220000: ("Northeast", "Northeast China"),
    230000: ("Northeast", "Northeast China"),
    # East China
    310000: ("East China", "East China"),
    320000: ("East China", "East China"),
    330000: ("East China", "East China"),
    340000: ("East China", "East China"),
    # South China
    350000: ("South China", "South China"),
    440000: ("South China", "South China"),
    450000: ("South China", "South China"),
    460000: ("South China", "South China"),
    # Central China
    410000: ("Central China", "Central China"),
    420000: ("Central China", "Central China"),
    430000: ("Central China", "Central China"),
    360000: ("Central China", "Central China"),
    # Southwest
    500000: ("Southwest", "Southwest China"),
    510000: ("Southwest", "Southwest China"),
    520000: ("Southwest", "Southwest China"),
    530000: ("Southwest", "Southwest China"),
    540000: ("Southwest", "Southwest China"),
    # Northwest China
    610000: ("Northwest China", "Northwest China"),
    620000: ("Northwest China", "Northwest China"),
    630000: ("Northwest China", "Northwest China"),
    640000: ("Northwest China", "Northwest China"),
    650000: ("Northwest China", "Northwest China"),
}

# ADCODE99 -> NAME_EN_JX (according to the table you gave)
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

# Read
gdf = gpd.read_file(province_no_TW_AM_HK_shp)
if "ADCODE99" not in gdf.columns:
    raise KeyError(f"Input shp is missing field ADCODE99. Existing fields:{list(gdf.columns)}")

gdf["_ADCODE99_INT"] = gdf["ADCODE99"].apply(to_int_adcode)

# Generate partition column (Shapefile field name <=10, so use DIV_CN / DIV_EN)
gdf["DIV_CN"] = gdf["_ADCODE99_INT"].map(lambda k: adcode_to_div.get(k, (None, None))[0])
gdf["DIV_EN"] = gdf["_ADCODE99_INT"].map(lambda k: adcode_to_div.get(k, (None, None))[1])

# Make sure NAME_EN_JX exists and is completed
if "NAME_EN_JX" not in gdf.columns:
    gdf["NAME_EN_JX"] = None
gdf["NAME_EN_JX"] = gdf["NAME_EN_JX"].fillna(gdf["_ADCODE99_INT"].map(adcode_to_name_en))

# Check not matched
unmatched = gdf[gdf["DIV_CN"].isna() | gdf["DIV_EN"].isna() | gdf["NAME_EN_JX"].isna()]
if len(unmatched) > 0:
    bad = sorted(set(unmatched["_ADCODE99_INT"].tolist()))
    raise ValueError(f"There is an unmatched/missing ADCODE99:{bad}. Please check whether the input shp contains abnormal records.")

# Only keep the columns you want (including NAME_EN_JX)
gdf_out = gdf[["ADCODE99", "NAME_EN_JX", "DIV_CN", "DIV_EN", "geometry"]].copy()

# Remove any existing output shapefile sidecar files before writing.
out_dir = os.path.dirname(province_no_TW_AM_HK_geographical_division_shp)
os.makedirs(out_dir, exist_ok=True)
base = os.path.splitext(province_no_TW_AM_HK_geographical_division_shp)[0]
for f in glob.glob(base + ".*"):
    try:
        os.remove(f)
    except Exception:
        pass

# Save
gdf_out.to_file(
    province_no_TW_AM_HK_geographical_division_shp,
    driver="ESRI Shapefile",
    encoding="UTF-8",  # For example, ArcGIS Chinese garbled characters can be changed to GBK
)

print("✅ Output completed:", province_no_TW_AM_HK_geographical_division_shp)
print("Output fields: ADCODE99, NAME_EN_JX, DIV_CN, DIV_EN, geometry")
