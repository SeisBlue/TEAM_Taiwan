import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.spatial import cKDTree

vs30_table = pd.read_csv(f"../data/Vs30ofTaiwan.csv")
TSMIP_station = pd.read_csv(f"../data/egdt_TSMIP_station_vs30.csv")

vs30_table["lon_binned"] = pd.cut(vs30_table["lon"], bins=30)  # 將經度分成 100 個區間
vs30_table["lat_binned"] = pd.cut(vs30_table["lat"], bins=40)  # 將緯度分成 100 個區間

# 使用 pivot_table 重塑數據，使經緯度成為網格的 x 和 y 軸
heatmap_data = vs30_table.pivot_table(
    index="lat_binned", columns="lon_binned", values="Vs30"
)

# 繪製熱力圖
plt.figure(figsize=(6, 8))
ax = sns.heatmap(heatmap_data, cmap="YlGnBu")
ax.invert_yaxis()

# 顯示圖形
plt.title("VS30 Heatmap")
plt.savefig("../data/VS30_heatmap.png")
plt.close()

plt.figure(figsize=(8, 8))
ax = sns.scatterplot(data=TSMIP_station, x="longitude", y="latitude", hue="Vs30", s=10)
plt.title("TSMIP Station Vs30")
plt.savefig("../data/TSMIP_station_vs30.png")

tree = cKDTree(vs30_table[["lat", "lon"]])
for i, row in TSMIP_station.iterrows():
    lon, lat = row["longitude"], row["latitude"]
    distance, index = tree.query([lat, lon])
    vs30 = vs30_table.iloc[index]["Vs30"]
    print(
        f"{row['station_code']} {lon}, {lat}: {row['Vs30']} get: {vs30}, err rate: {(row['Vs30']-vs30)/row['Vs30']}, method: {row['Vs30 reference']}"
    )
