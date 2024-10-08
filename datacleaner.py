# This is a sample Python script.

# Press Maiusc+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
import random

from pandas.plotting import boxplot


def hist_plot(df, col):
    sns.displot(df, x=col, kind="hist", row_order="desc", bins=15)
    plt.xticks(rotation=90, ha="right")
    plt.show()

def box_plot(df, x, y):
    sns.boxplot(x=x, y=y, data=df, palette='Set2')

    # Rotate the x-axis labels for better readability
    plt.xticks(rotation=90)

    # Set the plot title and labels
    plt.title('Birth Date Distribution by Nationality')
    plt.xlabel('Nationality')
    plt.ylabel('Birth Date')

    # Show the plot
    plt.tight_layout()
    plt.show()

def scatter_plot(df, x, y):
    sns.scatterplot(df, x=x, y=y)
    plt.xticks(rotation=90, ha="right")
    plt.show()

# Load the CSV file using pandas
class DataCleaner:
    def __init__(self, races_csv, cyclist_csv):
        self.races_df = pd.read_csv(races_csv, parse_dates=["date"])
        self.cyclist_df = pd.read_csv(cyclist_csv)
        self.df = pd.merge(
            self.races_df, self.cyclist_df, left_on="cyclist", right_on="_url", how="inner"
        )

        self.delete_column("_url_y")
        self.delete_column("name_y")
        self.delete_column("is_cobbled")
        self.delete_column("is_gravel")

        self.df.rename(
            columns={
                "name_x": "location",
                "_url_x": "_url",
            },
            inplace=True,
        )
        """
        self.df.rename(
            columns={
                "name_x": "Location",
                "profile": "Difficulty",
                "date": "Date",
                "cyclist": "Cyclist name",
                "cyclist_team": "Cyclist Team",
                "nationality": "Nationality",
                "points": "Primary points",
                "uci_points": "Secondary points",
                "length": "Circuit length",
                "position": "Arrival position",
                "climb_total": "Climb length",
                "cyclist_age": "Cyclist age",
                "delta": "Time from first",
                "birth_year": "Birth year",
                "weight": "Weight",
                "height": "Height",
                "startlist_quality": "Participants strength",
                "average_temperature": "Average temperature",
                "is_tarmac": "Is circuit on tarmac",
            },
            inplace=True,
        )"""

    def columns_names(self):
        return self.df.columns

    def delete_column(self, col):
        self.df.drop(columns=[col], inplace=True)

    def replace_NaN(self, column, value):
        self.df[column].fillna(value, inplace=True)

    # Loop through each column to get counts
    def inspect_for_missing(self):
        print(
            f"{'Column':<30} | {'Non-null count':<15} | {'Total count':<15} | {'Missing':<15}"
        )
        tmp = []
        for column in self.columns_names():
            non_null_count = self.df[
                column
            ].count()  # Count of non-null values
            total_count = len(self.df[column])  # Total number of values
            print(
                f"{column:<30} | {non_null_count:<15} | {total_count:<15} | {total_count != non_null_count}"
            )

            if total_count != non_null_count:
                tmp.append(column)

        return tmp

    def enumerate_column_range(self, col):
       return self.df[col].unique()

    def find_rows_with_alternatives(self, col1, col2):
        mask = self.df[col1].isna() ^ self.df[col2].isna()
        return self.df.loc[mask, 'name_x'].tolist()

    def check_are_alternatives(self, col1, col2):
        alternatives_rows = len(self.find_rows_with_alternatives(col1, col2))
        print(
            f"Columns: {col1}, {col2} "
            + f"{'YES. Columns are alternatives' if alternatives_rows == self.rows_count() else 'NO. Columns are not alternatives'}. "  # noqa
            + f"It's true only for {alternatives_rows}/{self.rows_count()} rows"
        )

    def get_birth_date_distributions(self):
        overall_distribution = self.df['birth_year'].describe()
        nationality_distributions = self.df.groupby('nationality')['birth_year'].describe()

        return overall_distribution, nationality_distributions

    def get_categorical_columns(self):
        return self.df.select_dtypes(
            include=["object", "category", "bool"]
        ).columns.tolist()

    def get_numerical_columns(self):
        return self.df.select_dtypes(include=["number"]).columns.tolist()

    def fix_missing_nationality(self):
        mask = self.df['nationality'].isna()
        # Only scott-davies does not have nationality => infer using wikipedia
        cyclists = set(self.df.loc[mask, 'cyclist'])
        for c in cyclists:
            mask = self.df["cyclist"] == c
            self.df.loc[mask, 'nationality'] = 'Britain'

    def reformat_date(self):
        for _, row in self.df.iterrows():
            tmp = datetime.strptime(row["date"], "%Y-%m-%d %H:%M:%S")
            new_tmp = tmp.replace(hour=0, minute=0, second=0)
            row["date"] = str(int(new_tmp.timestamp()))

    def rows_count(self):
        return len(self.df)
        
    def export_csv(self):
        return self.df.to_csv("./dataset/dataset.csv")
    


# categoricals_cols = dm.get_categorical_columns()
# numericals_cols = dm.get_numerical_columns()
# print(f"Categoricals columns: {categoricals_cols}")
# print(f"Numerical columns: {numericals_cols}")

dm = DataCleaner("./dataset/races.csv", "./dataset/cyclists.csv")
box_plot(dm.cyclist_df, 'nationality', 'birth_year')
v1, v2 = dm.get_birth_date_distributions()
print(v1)
print(v2)

# dm.hist_plot("is_tarmac")
# dm.hist_plot("is_cobbled")
# dm.hist_plot("is_gravel")
# print(dm.enumerate_column_range("is_tarmac"))
# dm.scatter_plot("Difficulty", "Primary points")
# dm.delete_column("Average temperature")

# dm.check_are_alternatives("is_cobbled", "is_gravel")

#dm = DataMiner(r"./dataset/races.csv", r"./dataset/cyclists.csv")
#dm.export_csv()



# print("END")
