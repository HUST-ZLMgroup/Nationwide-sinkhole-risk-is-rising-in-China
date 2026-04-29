import os
import pandas as pd
import geopandas as gpd

# =========================
# Basic variables and paths (as given by you)
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
# 1) CSV
# =========================
df = pd.read_csv(df_path)

if "ADCODE99" not in df.columns:
    raise KeyError(
        f"CSV is missing column ADCODE99. Existing columns:{list(df.columns)}"
    )

# , '110000' vs 110000.0
df["ADCODE99"] = pd.to_numeric(df["ADCODE99"], errors="coerce").astype("Int64")

# =========================
# 2) Read shp and extract the mapping table
# =========================
gdf_poly = gpd.read_file(province_no_TW_AM_HK_geographical_division_shp)

need_cols = {"ADCODE99", "NAME_EN_JX", "DIV_CN", "DIV_EN"}
missing = need_cols - set(gdf_poly.columns)
if missing:
    raise KeyError(f"SHP missing field:{missing}, existing fields:{list(gdf_poly.columns)}")

lookup = gdf_poly[["ADCODE99", "NAME_EN_JX", "DIV_CN", "DIV_EN"]].copy()
lookup["ADCODE99"] = pd.to_numeric(lookup["ADCODE99"], errors="coerce").astype("Int64")
lookup = lookup.drop_duplicates(subset=["ADCODE99"], keep="first")

# =========================
# 3) Merge and keep only matching lines
# =========================
out = df.merge(lookup, on="ADCODE99", how="left")

n_total = len(out)
n_matched = out["DIV_EN"].notna().sum()
n_dropped = n_total - n_matched

# (DIV_EN )
out_matched = out[out["DIV_EN"].notna()].copy()

print(f"Total number of lines:{n_total}")
print(f"Number of matched lines:{n_matched}")
print(f"Discard the number of unmatched rows:{n_dropped}")

# () ADCODE99,:
# bad_codes = sorted(out.loc[out["DIV_EN"].isna(), "ADCODE99"].dropna().unique().tolist())
# print("Discarded ADCODE99 (after deduplication):", bad_codes)
# =========================
# 4) Save
# =========================
os.makedirs(os.path.dirname(df_division_path), exist_ok=True)
out_matched.to_csv(df_division_path, index=False, encoding="utf-8-sig")
print("✅ Saved (matching lines only):", df_division_path)
