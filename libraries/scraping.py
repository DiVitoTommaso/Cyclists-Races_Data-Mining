import procyclingstats as pcs
import pandas as pd
from typing import List, Tuple, Union
from procyclingstats.errors import UnexpectedParsingError
from procyclingstats.table_parser import TableParser

# Function to convert a time string in the format 'HH:MM:SS' to seconds
def time_to_seconds(time):
    # Split the time string into hours, minutes, and seconds
    h, m, s = time.split(":")
    h = int(h) * 3600  # Convert hours to seconds
    m = int(m) * 60    # Convert minutes to seconds
    s = int(s)         # Seconds remain as is
    sign = -1 if m < 0 or h < 0 or s < 0 else 1  # Determine sign (positive or negative time)
    return sign * (abs(h) + abs(m) + abs(s))  # Return the total time in seconds with the correct sign

# Custom parse function for extracting specific fields from a table
def parse(self, fields: Union[List[str], Tuple[str, ...]]) -> None:
    # Initialize an empty list to store raw table rows
    raw_table = []
    for _ in range(self.table_length):
        raw_table.append({})

    # Loop through the fields to parse
    for field in fields:
        # Special handling for "class" field
        if field != "class":
            parsed_field_list = getattr(self, field)()  # Get the parsed values for the field
        else:
            parsed_field_list = getattr(self, "class_")()  # Handle "class" field separately

        # Check if the number of parsed values matches the number of rows in the table
        if len(parsed_field_list) != self.table_length:
            message = f"Field '{field}' wasn't parsed correctly"
            raise UnexpectedParsingError(message)  # Raise an error if parsing was incorrect

        # Assign the parsed values to the corresponding rows in the table
        for row, parsed_value in zip(raw_table, parsed_field_list):
            row[field] = parsed_value

    # Add all rows to the final table
    for row in raw_table:
        self.table.append(row)

    # Optional: Remove unwanted rows (commented out for now)
    # if "time" in fields and self.table:
    #     self._make_times_absolute()

# Assign the custom parse method to the TableParser class
TableParser.parse = parse

# Function to scrape and clean race results based on negative deltas
def scrape(df_races: pd.DataFrame) -> pd.DataFrame:
    # Get URLs of races with negative deltas
    bad_urls = df_races.loc[df_races["delta"] < 0, "_url"]
    bad_urls = bad_urls.unique()

    print("Scraping negative deltas")
    print("Negative deltas found in:")
    # Loop through each race with a negative delta
    for RACE_URL in bad_urls:
        stage = pcs.Stage(f"race/{RACE_URL}")  # Get the stage data from procyclingstats
        print(stage)  # Print stage information
        ranking = stage.results("rider_url", "time", "rank")  # Get the ranking data

        # Convert the ranking list to a pandas DataFrame
        df_ranking = pd.DataFrame(ranking)

        # Convert the "time" field to seconds using the time_to_seconds function
        df_ranking["time"] = df_ranking["time"].apply(time_to_seconds)

        # Get the first time (i.e., the winner's time)
        first_time = df_ranking["time"].loc[0]

        # Adjust negative times by adding them to the winner's time
        df_ranking["time"] = df_ranking["time"].apply(
            lambda x: x if x > 0 else first_time + x
        )

        # Set the winner's time (first place) to 0
        df_ranking.loc[0, "time"] = 0

        # Extract rider URLs and get the rider's unique ID (last part of the URL)
        df_ranking.rider_url = df_ranking.rider_url.apply(lambda x: x.split("/")[-1])

        # Loop through the rows of the ranking and update the delta values in df_races
        for i in range(len(df_ranking)):
            rider = df_ranking.loc[i, "rider_url"]
            time = df_ranking.loc[i, "time"]

            # Update the delta value for the rider in the df_races DataFrame
            df_races.loc[
                (df_races._url == RACE_URL) & (df_races.cyclist == rider),
                "delta",
            ] = time

    # Ensure that all delta values are positive integers after processing
    assert all(x.is_integer() for x in df_races.delta.dropna())
    return df_races

# Function to clean the delta values in the race data
def clean(df_races: pd.DataFrame) -> pd.DataFrame:
    # Make consistent deltas based on position (sorting and comparing adjacent rows)
    for stage in df_races._url.unique():
        df_stage = df_races.loc[df_races._url == stage]  # Get all rows for the current stage
        df_stage = df_stage.sort_values("position")  # Sort by position
        df_stage.reset_index(drop=True, inplace=True)  # Reset index after sorting

        # Loop through the stage results to ensure delta values are consistent
        for i in range(len(df_stage) - 1):
            if (
                df_stage.loc[i, "delta"] > df_stage.loc[i + 1, "delta"]
            ):  # Check if the next position has a faster time (lower delta)
                if (
                    df_stage.loc[i - 1, "delta"] <= df_stage.loc[i + 1, "delta"]
                ):  # If the next delta is consistent with previous, take average
                    df_races.loc[
                        (df_races._url == stage) & df_races.cyclist
                        == df_stage.loc[i, "cyclist"],
                        "delta",
                    ] = df_stage.loc[i, "delta"] = round(
                        (df_stage.loc[i - 1, "delta"] + df_stage.loc[i + 1, "delta"])
                        / 2
                    )
                else:  # If next delta is inconsistent, keep it as is
                    df_races.loc[
                        (df_races._url == stage) & df_races.cyclist
                        == df_stage.loc[i + 1, "cyclist"],
                        "delta",
                    ] = df_stage.loc[i + 1, "delta"] = df_stage.loc[i, "delta"]

        # Check that deltas are in increasing order (positions with higher deltas)
        assert all(x <= y for x, y in zip(df_stage.delta, df_stage.delta[1:]))

    return df_races

# Function to scrape race data with updated delta values (alternative version)
def scrape2(df_races: pd.DataFrame) -> pd.DataFrame:
    # Get URLs of races with negative deltas
    bad_urls = df_races.loc[df_races["delta"] < 0, "_url"]
    bad_urls = bad_urls.unique()

    print("Scraping negative deltas")
    print("Negative deltas found in:")
    # Loop through each race with a negative delta
    for RACE_URL in bad_urls:
        stage = pcs.Stage(f"race/{RACE_URL}")  # Get stage data from procyclingstats
        print(stage)  # Print stage info
        ranking = stage.results("rider_url", "time", "rank")  # Get ranking data

        # Convert ranking data to pandas DataFrame
        df_ranking = pd.DataFrame(ranking)

        # Get the first time (winner's time) for further delta calculations
        first_time = time_to_seconds(df_ranking["time"].loc[0])

        # Set the winner's time to 0
        df_ranking.loc[0, "time"] = 0

        # Extract rider URLs to get the rider's unique ID
        df_ranking.rider_url = df_ranking.rider_url.apply(lambda x: x.split("/")[-1])

        # Loop through each rider in the ranking and update delta values in df_races
        for i in range(len(df_ranking)):
            rider = df_ranking.loc[i, "rider_url"]
            mask = (df_races._url == RACE_URL) & (df_races.cyclist == rider)  # Mask for current race and rider
            try:
                delta_value = df_races.loc[mask, "delta"].values[0]  # Get the delta value for the rider
            except IndexError:
                continue  # Skip if the rider is not found
            # Update the delta value, adding the time difference to the winner's time
            df_races.loc[
                mask,
                "delta",
            ] = (
                first_time + delta_value if delta_value < 0 else delta_value
            )

    # Ensure that delta contains only positive floats
    assert all(x.is_integer() for x in df_races.delta.dropna())
    return df_races
