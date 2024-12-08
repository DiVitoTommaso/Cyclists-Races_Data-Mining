import procyclingstats as pcs
import pandas as pd
from typing import List, Tuple, Union
from procyclingstats.errors import UnexpectedParsingError
from procyclingstats.table_parser import TableParser


def time_to_seconds(time):
    h, m, s = time.split(":")
    h = int(h) * 3600
    m = int(m) * 60
    s = int(s)
    sign = -1 if m < 0 or h < 0 or s < 0 else 1
    return sign * (abs(h) + abs(m) + abs(s))


def parse(self, fields: Union[List[str], Tuple[str, ...]]) -> None:

    raw_table = []
    for _ in range(self.table_length):
        raw_table.append({})

    for field in fields:
        if field != "class":
            parsed_field_list = getattr(self, field)()
        # special case when field is called class
        else:
            parsed_field_list = getattr(self, "class_")()
        # field wasn't found in every table row, so isn't matching table
        # rows correctly
        if len(parsed_field_list) != self.table_length:
            message = f"Field '{field}' wasn't parsed correctly"
            raise UnexpectedParsingError(message)

        for row, parsed_value in zip(raw_table, parsed_field_list):
            row[field] = parsed_value

    # remove unwanted rows
    for row in raw_table:
        self.table.append(row)

    # if "time" in fields and self.table:
    # self._make_times_absolute()


TableParser.parse = parse


def scrape(df_races: pd.DataFrame) -> pd.DataFrame:

    # get _url which have negative delta
    bad_urls = df_races.loc[df_races["delta"] < 0, "_url"]
    bad_urls = bad_urls.unique()

    print("Scraping negative deltas")
    print("Negative deltas found in:")
    for RACE_URL in bad_urls:

        stage = pcs.Stage(f"race/{RACE_URL}")
        print(stage)
        ranking = stage.results("rider_url", "time", "rank")
        # for i in sorted(ranking,key = lambda x: x['rank']):
        #    print(i)

        # convert ranking to pandas table, ranking is a list of objects
        df_ranking = pd.DataFrame(ranking)

        df_ranking["time"] = df_ranking["time"].apply(time_to_seconds)

        # first time is the time of the winner
        first_time = df_ranking["time"].loc[0]

        # sum first time to all other negative times
        df_ranking["time"] = df_ranking["time"].apply(
            lambda x: x if x > 0 else first_time + x
        )

        df_ranking.loc[0, "time"] = 0

        df_ranking.rider_url = df_ranking.rider_url.apply(lambda x: x.split("/")[-1])

        for i in range(len(df_ranking)):
            rider = df_ranking.loc[i, "rider_url"]
            time = df_ranking.loc[i, "time"]

            df_races.loc[
                (df_races._url == RACE_URL) & (df_races.cyclist == rider),
                "delta",
            ] = time

        # print(
        #     df_races.loc[
        #         (df_races._url == RACE_URL),
        #         "delta",
        #     ]
        # )

    # check if delta contains positive floats
    assert all(x.is_integer() for x in df_races.delta.dropna())
    return df_races


def clean(df_races: pd.DataFrame) -> pd.DataFrame:
    # make consistent deltas with position
    # COMMENT THIS PORTION OF CODE TO OBTAIN DELTAS AFTER CLEANING
    # AND BEFORE IMPUTATION
    for stage in df_races._url.unique():
        df_stage = df_races.loc[df_races._url == stage]
        df_stage = df_stage.sort_values("position")
        df_stage.reset_index(drop=True, inplace=True)

        for i in range(len(df_stage) - 1):
            if (
                df_stage.loc[i, "delta"] > df_stage.loc[i + 1, "delta"]
            ):  # next position is faster
                if (
                    df_stage.loc[i - 1, "delta"] <= df_stage.loc[i + 1, "delta"]
                ):  # next consistent with previous -> put as average
                    df_races.loc[
                        (df_races._url == stage) & df_races.cyclist
                        == df_stage.loc[i, "cyclist"],
                        "delta",
                    ] = df_stage.loc[i, "delta"] = round(
                        (df_stage.loc[i - 1, "delta"] + df_stage.loc[i + 1, "delta"])
                        / 2
                    )
                else:  # next not consistent with previous -> put as present
                    df_races.loc[
                        (df_races._url == stage) & df_races.cyclist
                        == df_stage.loc[i + 1, "cyclist"],
                        "delta",
                    ] = df_stage.loc[i + 1, "delta"] = df_stage.loc[i, "delta"]

        # control every position has higher delta
        assert all(x <= y for x, y in zip(df_stage.delta, df_stage.delta[1:]))

    return df_races
