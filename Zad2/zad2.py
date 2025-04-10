import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import zscore
import os
import pandas as pd
from google.cloud import bigquery
os.environ[
    "GOOGLE_APPLICATION_CREDENTIALS"] = "valiant-circuit-453011-d0-0c253bd2ed73.json"
client = bigquery.Client()

#CZĘŚC 2

df = pd.read_csv("results/covid_5_1.csv")
df2 = pd.read_csv("results/covid_5_2.csv")
df3 = pd.read_csv("results/part5bigdata.csv")

df_zscore = df.copy()
df_iqr = df2.copy()

# Z-SCORE dla 'new_confirmed'

z_scores = zscore(df_zscore["new_confirmed"])
abs_z_scores = np.abs(z_scores)

threshold = 2
filtered_entries = abs_z_scores < threshold

xmin = df["new_confirmed"].min()
xmax = df["new_confirmed"].max()

# Usuwamy outliery
df_zscore = df_zscore[filtered_entries]

# Wykres przed i po
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
sns.boxplot(x=df["new_confirmed"])
plt.title("Przed usunięciem (new_confirmed)")
plt.xlim(xmin, xmax)

plt.subplot(1, 2, 2)
sns.boxplot(x=df_zscore["new_confirmed"])
plt.title("Po Z-Score (new_confirmed)")
plt.tight_layout()
plt.xlim(xmin, xmax)
plt.show()

# IQR dla 'gdp_usd'
Q1 = df_iqr["gdp_usd"].quantile(0.25)
Q3 = df_iqr["gdp_usd"].quantile(0.75)
IQR = Q3 - Q1

# Granice
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Filtracja
df_iqr = df_iqr[(df_iqr["gdp_usd"] >= lower_bound) & (df_iqr["gdp_usd"] <= upper_bound)]

# Zakresy osi x na podstawie oryginalnych danych
xmin = df2["gdp_usd"].min()
xmax = df2["gdp_usd"].max()

# Wykres przed i po
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
sns.boxplot(x=df2["gdp_usd"])
plt.title("Przed usunięciem (GDP_usd)")
plt.xlim(xmin, xmax)  # ustalenie zakresu osi x

plt.subplot(1, 2, 2)
sns.boxplot(x=df_iqr["gdp_usd"])
plt.title("Po IQR (GDP_usd)")
plt.xlim(xmin, xmax)  # ten sam zakres osi x dla porównania

plt.tight_layout()
plt.show()

# Z-SCORE dla 'gdp_usd'
df_zscore_gdp = df2.copy()

z_scores_gdp = zscore(df_zscore_gdp["gdp_usd"])
abs_z_scores_gdp = np.abs(z_scores_gdp)

threshold = 2
filtered_entries_gdp = abs_z_scores_gdp < threshold

# Zakresy osi x przed filtrowaniem
xmin = df2["gdp_usd"].min()
xmax = df2["gdp_usd"].max()

df_zscore_gdp = df_zscore_gdp[filtered_entries_gdp]

# Wykresy
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
sns.boxplot(x=df2["gdp_usd"])
plt.title("GDP_usd - przed (Z-Score)")
plt.xlim(xmin, xmax)

plt.subplot(1, 2, 2)
sns.boxplot(x=df_zscore_gdp["gdp_usd"])
plt.title("GDP_usd - po (Z-Score)")
plt.xlim(xmin, xmax)

plt.tight_layout()
plt.show()


#new_confirmed z IQR

df_iqr_cases = df.copy()

Q1 = df_iqr_cases["new_confirmed"].quantile(0.25)
Q3 = df_iqr_cases["new_confirmed"].quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR


df_iqr_cases = df_iqr_cases[
    (df_iqr_cases["new_confirmed"] >= lower_bound) &
    (df_iqr_cases["new_confirmed"] <= upper_bound)
]

# Zakresy osi x na podstawie oryginalnych danych
xmin = df["new_confirmed"].min()
xmax = df["new_confirmed"].max()

plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
sns.boxplot(x=df["new_confirmed"])
plt.title("new_confirmed - przed IQR")
plt.xlim(xmin, xmax)

plt.subplot(1, 2, 2)
sns.boxplot(x=df_iqr_cases["new_confirmed"])
plt.title("new_confirmed - po IQR")
plt.xlim(xmin, xmax)

plt.tight_layout()
plt.show()

# CZĘŚC 5

def show_corr_matrix(df, cols, title):
    corr = df[cols].corr()
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", square=True)
    plt.title(title)
    plt.show()

cols_5_1 = ["new_confirmed", "new_persons_vaccinated", "new_deceased"]
show_corr_matrix(df, cols_5_1, "Korelacja: Nowe przypadki, szczepienia i zgony")

cols_5_2 = ["new_confirmed", "new_persons_vaccinated", "new_deceased","cumulative_confirmed",
            "cumulative_persons_fully_vaccinated", "cumulative_deceased"]
show_corr_matrix(df, cols_5_2, "Korelacja: Nowe vs Łączne przypadki, szczepienia, zgony")

cols_5_3 = ["gdp_per_capita_usd", "human_development_index", "new_confirmed", "new_persons_vaccinated", "new_deceased"]
show_corr_matrix(df3, cols_5_3, "Korelacja: Wskaźniki gospodarcze vs nowe przypadki, szczepienia i zgony")

cols_5_4 = ["population", "population_urban", "population_density", "new_confirmed", "new_persons_vaccinated", "new_deceased"]
show_corr_matrix(df3, cols_5_4, "Korelacja: Parametry demograficzne vs nowe przypadki, szczepienia i zgony")

cols_5_5 = ['population', "new_confirmed", "new_persons_vaccinated", "new_deceased"]
show_corr_matrix(df3, cols_5_5, "Korelacja: Gęstość zaludnienia vs nowe przypadki, szczepienia i zgony")

cols_5_6 = ["new_confirmed", "new_persons_vaccinated", "new_deceased", "population_density"]
show_corr_matrix(df3, cols_5_6, "Korelacja: nowe przypadki, szczepienia i zgony vs Gęstość zaludnienia")

cols_5_7= ["human_development_index", "new_confirmed", "new_persons_vaccinated", "new_deceased"]
show_corr_matrix(df3, cols_5_7, "Korelacja: HDI vs nowe przypadki, szczepienia i zgony")

cols_5_8= ['gdp_usd', "new_confirmed", "new_persons_vaccinated", "new_deceased"]
show_corr_matrix(df3, cols_5_8, "Korelacja: PKB vs nowe przypadki, szczepienia i zgony")




# query = 'select iso_3166_1_alpha_3, country_name , population, new_deceased, cumulative_persons_vaccinated, new_persons_vaccinated, cumulative_persons_fully_vaccinated, new_persons_fully_vaccinated, cumulative_deceased, population_density, new_confirmed, cumulative_confirmed, population_urban, human_development_index, area_sq_km, gdp_usd, gdp_per_capita_usd, area_sq_km from bigquery-public-data.covid19_open_data.covid19_open_data where aggregation_level = 0 and population is not null and iso_3166_1_alpha_3 is not null and area_sq_km is not null'
# df3 = client.query(query).result().to_dataframe()
# df3.dropna(subset=['population', 'iso_3166_1_alpha_3'], inplace=True)
#
# columns_to_convert = ['iso_3166_1_alpha_3', 'country_name']
# df3[columns_to_convert] = df3[columns_to_convert].astype("string")
# cols_to_int = ['cumulative_confirmed', 'new_confirmed', 'new_deceased', 'cumulative_deceased', 'cumulative_persons_vaccinated', 'new_persons_vaccinated', 'cumulative_persons_fully_vaccinated', 'new_persons_fully_vaccinated']
# df3[cols_to_int] = df3[cols_to_int].apply(pd.to_numeric, errors='coerce').astype('Int64')
# columns_to_check = ['new_confirmed', 'cumulative_confirmed','new_deceased', 'cumulative_deceased']
# df3.dropna(subset=columns_to_check, how='all', inplace=True)
# df3[cols_to_int] = df3[cols_to_int].fillna(0)
# df3['population'] = pd.to_numeric(df3['population'], errors='coerce')
# df3['area_sq_km'] = pd.to_numeric(df3['area_sq_km'], errors='coerce')
# df3['population_density'] = pd.to_numeric(df3['population_density'], errors='coerce')
# df3['area_sq_km'] = pd.to_numeric(df3['area_sq_km'], errors='coerce')
# df3['population_urban'] = pd.to_numeric(df3['population_urban'], errors='coerce')
# df3['human_development_index'] = pd.to_numeric(df3['human_development_index'], errors='coerce')
# df3['gdp_per_capita_usd'] = pd.to_numeric(df3['gdp_per_capita_usd'], errors='coerce')
# df3['gdp_usd'] = pd.to_numeric(df3['gdp_usd'], errors='coerce')
# cols_to_0 = [ 'population_density', 'gdp_usd']
# df3[cols_to_0] = df3[cols_to_0].fillna(0)
#
# df3.to_csv('results/part5bigdata.csv', index=False)