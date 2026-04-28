import os
import pandas as pd
import geopandas as gpd

# =========================
# 基础变量与路径（按你给的）
# =========================
home_dir = os.path.expanduser("/path/to/home/")
base_path = os.path.join(home_dir, "PROJECT_DIR", "Papers", "4_NC")
data_base_path = os.path.join(base_path, "data")

Extracted_HAVE_future = "Extracted_HAVE_future"
Positive_Negative_balanced = "Positive_Negative_balanced"

df_path = os.path.join(
    data_base_path, Extracted_HAVE_future, Positive_Negative_balanced,
    "AllFeatures_Positive_Negative_balanced_25366_ssp1_cleaned.csv"
)

province_no_TW_AM_HK_geographical_division_shp = os.path.join(
    base_path, "data", "Administrative_divisions_of_china", "no_TW_AM_HK",
    "china_pro2_no_TW_AM_HK_geographical_division.shp"
)

df_division_path = os.path.join(
    data_base_path, Extracted_HAVE_future, Positive_Negative_balanced,
    "AllFeatures_Positive_Negative_balanced_25366_ssp1_cleaned_division.csv"
)

# =========================
# 1) 读取 CSV
# =========================
df = pd.read_csv(df_path)

if "ADCODE99" not in df.columns:
    raise KeyError(
        f"CSV 缺少 ADCODE99 列。现有列：{list(df.columns)}"
    )

# 统一为可空整型，避免 '110000' vs 110000.0 匹配失败
df["ADCODE99"] = pd.to_numeric(df["ADCODE99"], errors="coerce").astype("Int64")

# =========================
# 2) 读取 shp，提取映射表
# =========================
gdf_poly = gpd.read_file(province_no_TW_AM_HK_geographical_division_shp)

need_cols = {"ADCODE99", "NAME_EN_JX", "DIV_CN", "DIV_EN"}
missing = need_cols - set(gdf_poly.columns)
if missing:
    raise KeyError(f"SHP 缺少字段：{missing}，现有字段：{list(gdf_poly.columns)}")

lookup = gdf_poly[["ADCODE99", "NAME_EN_JX", "DIV_CN", "DIV_EN"]].copy()
lookup["ADCODE99"] = pd.to_numeric(lookup["ADCODE99"], errors="coerce").astype("Int64")
lookup = lookup.drop_duplicates(subset=["ADCODE99"], keep="first")

# =========================
# 3) 合并并仅保留匹配到的行
# =========================
out = df.merge(lookup, on="ADCODE99", how="left")

n_total = len(out)
n_matched = out["DIV_EN"].notna().sum()
n_dropped = n_total - n_matched

# 只保留匹配到分区的行（DIV_EN 不为空即可）
out_matched = out[out["DIV_EN"].notna()].copy()

print(f"总行数：{n_total}")
print(f"匹配到的行数：{n_matched}")
print(f"丢弃未匹配行数：{n_dropped}")

# （可选）如果你还想看看丢弃的是哪些 ADCODE99，取消注释：
# bad_codes = sorted(out.loc[out["DIV_EN"].isna(), "ADCODE99"].dropna().unique().tolist())
# print("被丢弃的 ADCODE99（去重后）：", bad_codes)

# =========================
# 4) 保存
# =========================
os.makedirs(os.path.dirname(df_division_path), exist_ok=True)
out_matched.to_csv(df_division_path, index=False, encoding="utf-8-sig")
print("✅ 已保存（仅匹配行）：", df_division_path)
