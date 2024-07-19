import pandas as pd
from io_utils import no_geo_to_csv, load_dataframes
from dataframe_utils import compare_ner_2_geo, extract_places


def comparison_result_dict(twitter_df, ner_df_list):
    tweets_dataframe = twitter_df
    ner_list = ner_df_list
    ner_with_places = ner_list[0]

    result = [
        compare_ner_2_geo(x, y, z)
        for x, y, z in zip(
            ner_with_places["tweet_id"],
            ner_with_places["place_name"],
            ner_with_places["full_name"],
        )
    ]
    tweet_id_target_data = {
        "tweet_id": [tweet_id[0] for tweet_id in result],
        "target": [target[1] for target in result],
    }
    dataframe_target = pd.DataFrame(tweet_id_target_data)
    dataframe_target = dataframe_target.set_index("tweet_id")
    ner_with_places = ner_with_places.set_index("tweet_id")
    ner_and_target = ner_with_places.join(dataframe_target, on="tweet_id", how="inner")

    # Variables numéricas para devolver
    ner_with_places_count = len(ner_with_places.index)
    ner_without_places_count = len(ner_list[1].index)
    failed_tweets = len(ner_and_target.loc[ner_and_target.target == 0])
    correct_tweets = len(ner_and_target.loc[ner_and_target.target == 1])
    total_tweets = len(tweets_dataframe.index)

    result_dict = {
        "total_tweets": total_tweets,
        "ner_with_places_count": ner_with_places_count,
        "ner_without_places_count": ner_without_places_count,
        "failed_tweets": failed_tweets,
        "correct_tweets": correct_tweets,
        "accuracy_percentage": (correct_tweets * 100) / ner_with_places_count,
        "ner_and_target": ner_and_target,
    }

    return result_dict


def comparison_result_to_string(result_dict):
    return (
        "\nCantidad de tweets analizados: ",
        result_dict["total_tweets"],
        "\nCantidad de tweets con entidades reconocibles de lugar: ",
        result_dict["ner_with_places_count"] + result_dict["ner_without_places_count"],
        "\nCantidad de tweets con localización activada: ",
        result_dict["ner_with_places_count"],
        "\nCantidad de tweets con localización desactivada: ",
        result_dict["ner_without_places_count"],
        "\nCantidad de tweets acertados (localización activada): ",
        result_dict["correct_tweets"],
        "\nCantidad de tweets no acertados (localización activada): ",
        result_dict["failed_tweets"],
        "\nPorcentaje de acierto (localización activada): {0:.2f}".format(
            result_dict["accuracy_percentage"]
        ),
        "%",
    )


# Cargar dataframes y pasar datos desde aca, poner como parámetro también en las funciones
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
