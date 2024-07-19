import pandas as pd
from io_utils import no_geo_to_csv, load_dataframes
from dataframe_utils import extract_places, comparison_result_dict
from parsing_utils import comparison_result_to_string


def main():
    raw_dataframes_list = load_dataframes()
    places_df = raw_dataframes_list[0]
    twitter_df = raw_dataframes_list[1]

    ner_tweets_dataframe_list = extract_places(places_df, twitter_df)
    ner_without_geo = ner_tweets_dataframe_list[1]

    comparison_dict = comparison_result_dict(twitter_df, ner_tweets_dataframe_list)
    print(comparison_result_dict(comparison_dict))
    no_geo_to_csv(ner_without_geo, twitter_df)


main()
