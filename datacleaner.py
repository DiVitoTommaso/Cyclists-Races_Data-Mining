# This is a sample Python script.

# Press Maiusc+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime

def hist_plot(df, col):
    sns.displot(df, x=col, kind="hist", row_order="desc", bins=15)
    plt.xticks(rotation=90, ha="right")
    plt.show()

def box_plot(df, x, y):
    sns.boxplot(x=x, y=y, data=df, palette='Set2')

    # Rotate the x-axis labels for better readability
    plt.xticks(rotation=90)

    # Set the plot title and labels
    plt.title('Distribution')
    plt.xlabel(x)
    plt.ylabel(y)

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
        # So keep ref to all datasets and make one joined
        self.races_df = pd.read_csv(races_csv, parse_dates=["date"])
        self.cyclist_df = pd.read_csv(cyclist_csv)
        self.df = pd.merge(
            self.races_df, self.cyclist_df, left_on="cyclist", right_on="_url", how="inner"
        )

        # Delete useless columns
        self.delete_column("_url_y")
        self.delete_column("name_y")

        # Delete columns that are all false
        self.delete_column("is_cobbled")
        self.delete_column("is_gravel")

        # Rename to understand better
        self.df.rename(
            columns={
                "name_x": "location",
                "_url_x": "id",
            },
            inplace=True,
        )

    # Delete a column
    def delete_column(self, col):
        self.df.drop(columns=[col], inplace=True)

    # Replace NaN in specied column with default value
    def replace_NaN(self, column, value):
        self.df[column].fillna(value, inplace=True)

    # Get information from the dataset
    def describe(self):
        print(
            f"{'Column':<30} | {'Non-null count':<15} | {'Total count':<15} | {'Missing':<15}"
        )
        tmp = []
        total = len(self.df)  # Total number of values
        for column in self.df.columns:
            non_null_count = self.df[column].count()  # Count of non-null values
            print(f"{column:<30} | {non_null_count:<15} | {total:<15} | {total != non_null_count}"
            )

            if total != non_null_count:
                tmp.append(column)

        return tmp

    # Get range of values of a column
    def enumerate_column_range(self, col):
       return self.df[col].unique()

    # Check if columns are mutually exclusive
    def are_xor_columns(self, col1, col2):
        mask = self.df[col1].isna() ^ self.df[col2].isna()
        res = self.df.loc[mask, 'name_x'].tolist()
        return len(res) == len(self.df)

    # Get distribution of birth year overall and by nationality to fix missing values
    def get_birth_distributions(self):
        overall_distribution = self.cyclist_df['birth_year'].describe()
        nationality_distributions = self.cyclist_df.groupby('nationality')['birth_year'].describe()

        return overall_distribution, nationality_distributions

    # Get distribution of age overall and by nationality to fix missing values. Really needed?
    def get_age_distributions(self):
        overall_distribution = self.cyclist_df.groupby('cyclist_age')['cyclist_age'].describe()
        nationality_distributions = self.cyclist_df.groupby(['cyclist_age', 'nationality'])['cyclist_age'].describe()

        return overall_distribution, nationality_distributions

    # Get distribution of age overall and by nationality to fix missing values
    def get_height_distributions(self):
        overall_distribution = self.cyclist_df['height'].describe()
        nationality_distributions = self.cyclist_df.groupby('nationality')['height'].describe()

        return overall_distribution, nationality_distributions

    # Get distribution of weight overall and by nationality to fix missing values
    def get_weight_distributions(self):
        overall_distribution = self.cyclist_df['weight'].describe()
        nationality_distributions = self.cyclist_df.groupby('nationality')['weight'].describe()

        return overall_distribution, nationality_distributions

    # Assume NaN == No team. Cyclist rode without a team
    def fix_missing_team(self):
        self.replace_NaN('cyclist_team', 'No team')

    # How can we fix it? No climb ==> ??? Some profiles with 'hilly' or 'mountains' has nan
    def get_climb_distributions(self):
        profile_distributions = self.cyclist_df.groupby('profile')['climb_total'].describe()

        return profile_distributions

    def get_numerical_columns(self):
        return self.df.select_dtypes(include=["number"]).columns.tolist()

    def get_categorical_columns(self):
        return self.df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

    # Fix cyclist with missing nationality and birth year (scott-davies) using wikipedia
    def fix_cyclist(self):
        mask = self.df['nationality'].isna()
        cyclists = set(self.df.loc[mask, 'cyclist'])
        for c in cyclists:
            mask = self.df["cyclist"] == c
            self.df.loc[mask, 'nationality'] = 'Britain'
            self.df.loc[mask, 'birth_year'] = 1995
            for _, row in self.df.iterrows():
                row['cyclist_age'] = datetime.strptime(row["date"], "%Y-%m-%d %H:%M:%S") - datetime.strptime("1995-08-05", "%Y-%m-%d")

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
print(f"{sorted(dm.cyclist_df['weight'].unique())}")
#box_plot(dm.cyclist_df, 'nationality', 'birth_year')
#v1, v2 = dm.get_birth_date_distributions()
#print(v1)
#print(v2)

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
