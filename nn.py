import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 11

df = pd.read_csv("data/netflix_titles.csv")
print("Rows and columns:", df.shape)
df.head()

df.info()

# Missing values per column
missing = df.isnull().sum().sort_values(ascending=False)
missing = missing[missing > 0]
missing

df_clean = df.copy()

for col in ['director', 'cast', 'country']:
    df_clean[col] = df_clean[col].fillna('Unknown')

df_clean = df_clean.dropna(subset=['date_added', 'rating'])

df_clean['date_added'] = pd.to_datetime(df_clean['date_added'].str.strip(), format='mixed')
df_clean['year_added'] = df_clean['date_added'].dt.year
df_clean['month_added'] = df_clean['date_added'].dt.month

print("Rows after cleaning:", df_clean.shape[0])
df_clean.head()

type_counts = df_clean['type'].value_counts()
print(type_counts)

fig, ax = plt.subplots(1, 2, figsize=(14, 5))

ax[0].bar(type_counts.index, type_counts.values, color=['#E50914', '#221f1f'])
ax[0].set_title('Number of Movies vs. TV Shows')
ax[0].set_ylabel('Count')
for i, v in enumerate(type_counts.values):
    ax[0].text(i, v + 20, str(v), ha='center', fontweight='bold')

ax[1].pie(type_counts.values, labels=type_counts.index, autopct='%1.1f%%',
          colors=['#E50914', '#221f1f'], startangle=90)
ax[1].set_title('Percentage Share')

plt.tight_layout()
plt.show()

yearly = df_clean.groupby(['year_added', 'type']).size().unstack(fill_value=0)

fig, ax = plt.subplots(figsize=(12, 6))
yearly.plot(kind='line', marker='o', ax=ax, color=['#E50914', '#221f1f'])
ax.set_title('Titles Added Per Year')
ax.set_xlabel('Year')
ax.set_ylabel('Number of Titles Added')
ax.legend(title='Type')
plt.tight_layout()
plt.show()

# Some rows list multiple countries separated by commas; we take the first one for simplicity
top_countries = (
    df_clean[df_clean['country'] != 'Unknown']['country']
    .apply(lambda x: x.split(',')[0].strip())
    .value_counts()
    .head(10)
)

fig, ax = plt.subplots(figsize=(10, 6))
sns.barplot(x=top_countries.values, y=top_countries.index, hue=top_countries.index,
            palette='Reds_r', legend=False, ax=ax)
ax.set_title('Top 10 Countries Producing Content on Netflix')
ax.set_xlabel('Number of Titles')
ax.set_ylabel('Country')
plt.tight_layout()
plt.show()

from collections import Counter

genres = df_clean['listed_in'].str.split(', ').explode()
top_genres = Counter(genres).most_common(10)
genres_df = pd.DataFrame(top_genres, columns=['genre', 'count'])

fig, ax = plt.subplots(figsize=(10, 6))
sns.barplot(data=genres_df, x='count', y='genre', hue='genre', palette='mako', legend=False, ax=ax)
ax.set_title('Top 10 Most Common Genres')
ax.set_xlabel('Number of Titles')
ax.set_ylabel('Genre')
plt.tight_layout()
plt.show()

movies = df_clean[df_clean['type'] == 'Movie'].copy()
movies['duration_min'] = movies['duration'].str.replace(' min', '', regex=False)
movies['duration_min'] = pd.to_numeric(movies['duration_min'], errors='coerce')
movies = movies.dropna(subset=['duration_min'])

fig, ax = plt.subplots(figsize=(10, 6))
sns.histplot(movies['duration_min'], bins=30, kde=True, color='#E50914', ax=ax)
ax.axvline(movies['duration_min'].mean(), color='black', linestyle='--',
           label=f"Mean: {movies['duration_min'].mean():.0f} min")
ax.set_title('Distribution of Movie Durations')
ax.set_xlabel('Duration (minutes)')
ax.set_ylabel('Number of Movies')
ax.legend()
plt.tight_layout()
plt.show()