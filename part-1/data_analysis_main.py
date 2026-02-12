from pathlib import Path

import matplotlib
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

matplotlib.use('TkAgg')


CURRENT_FILE = Path(__file__).resolve()

PROJECT_ROOT = CURRENT_FILE.parent.parent

FILE_PATH = PROJECT_ROOT / "players_22.csv"

DATA_FRAME = None


# Loading data from CSV file
def load_data(file_name):
    data_frame = pd.read_csv(file_name, low_memory=False)
    return data_frame


# Preparing preliminary statistics and saving them to file
def preliminary_statistics():

    # Numeral traits statistics
    df_numeric = DATA_FRAME.select_dtypes(include=['number'])
    numeric_stats = df_numeric.describe(percentiles=[0.05, 0.95])
    numeric_stats['missing_values'] = df_numeric.isnull().sum()

    # Categorial traits statistics
    df_categorical = DATA_FRAME.select_dtypes(include=['object'])
    categorical_stats = df_categorical.nunique().to_frame(name='unique_values')
    categorical_stats['missing_values'] = df_categorical.isnull().sum()
    categorical_stats['class_proportion'] = df_categorical.apply(lambda x: x.value_counts(normalize=True).to_dict())

    # Saving statistics to CSV files
    numeric_stats.to_csv('numeric_stats.csv')
    categorical_stats.T.to_csv('categorical_stats.csv')


def draw_a_boxplot(trait):

    sns.boxplot(data=DATA_FRAME, x=trait)
    plt.show()


def draw_a_violinplot(trait):

    sns.violinplot(data=DATA_FRAME, x=trait)
    plt.show()


def draw_an_error_bar():

    new_data_frame = DATA_FRAME[['player_positions', 'height_cm']].dropna()
    new_data_frame['main_position'] = new_data_frame['player_positions'].apply(lambda x: x.split(',')[0])
    position_stats = new_data_frame.groupby('main_position')['height_cm'].agg(['mean', 'std']).reset_index()

    plt.figure(figsize=(10, 6))
    ax = sns.barplot(data=position_stats, x='main_position', y='mean', hue="main_position", palette='viridis',
                     legend=False)
    plt.errorbar(x=range(len(position_stats)), y=position_stats['mean'],
                 yerr=position_stats['std'], fmt='none', capsize=5, color='black', elinewidth=2)

    plt.xlabel('Pozycja na boisku')
    plt.ylabel('Średni wzrost (cm)')
    plt.title('Średni wzrost graczy na różnych pozycjach')
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    plt.show()


def draw_an_histogram(trait):

    sns.displot(x=DATA_FRAME[trait])

    plt.show()


def draw_a_conditional_histogram(trait1, trait2):

    sns.displot(DATA_FRAME, x=trait1, hue=trait2)

    plt.show()


def draw_a_heatmap(trait1, trait2, trait3):

    df_pivot = DATA_FRAME.pivot_table(index=trait1, columns=trait2, values=trait3, aggfunc='mean')

    sns.heatmap(df_pivot)

    plt.show()


def draw_an_estimated_regression_fit(trait1, trait2):
    
    sns.regplot(x=trait1, y=trait2, data=DATA_FRAME, scatter_kws={"s": 5});

    plt.show()


if __name__ == '__main__':

    DATA_FRAME = load_data(FILE_PATH)

    if DATA_FRAME is not None:

        # 3.0
        preliminary_statistics()

        # 3.5
        draw_a_boxplot('overall')
        draw_a_violinplot('weight_kg')

        # 4.0
        draw_an_error_bar()
        draw_an_histogram('age')
        draw_a_conditional_histogram('shooting', 'preferred_foot')

        # 4.5
        draw_a_heatmap('height_cm', 'weight_kg', 'pace')

        # 5.0
        draw_an_estimated_regression_fit('overall', 'value_eur')
