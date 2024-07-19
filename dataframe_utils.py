from ner_utils import predicted_with_id
import pandas as pd


# Devuelve una lista, con el dataframe de las entidades extraídas con geo, y las sin
def extract_places(places_df, twitter_df):
    twitter_df_limited = twitter_df

    # Proceso los datos, solo tomando las columnas de texto, id y localización
    result = [
        predicted_with_id(x, y, z)
        for x, y, z in zip(
            twitter_df_limited["text"],
            twitter_df_limited["id"],
            twitter_df_limited["geo"],
        )
    ]
    # Descarto aquellos que no contaban con ninguna entidad de localización reconocible
    result = list(filter(lambda item: item is not None, result))
    # Guardo los datos en un diccionario para luego convertirlo en un DataFrame
    ner_tweets = {
        "tweet_id": [tweet_id[0] for tweet_id in result],
        "place_name": [place_name[1] for place_name in result],
        "place_id": [place_id[2] for place_id in result],
    }
    ner_tweets_df = pd.DataFrame(data=ner_tweets)
    # Los que no tienen localización, los llevo a otro DF
    ner_tweets_no_geo_df = ner_tweets_df.loc[ner_tweets_df["place_id"].isnull()]
    ner_tweets_df = ner_tweets_df.loc[ner_tweets_df["place_id"].notnull()]
    # Join de los dos dataframes, mediante los id de lugar
    ner_with_places = ner_tweets_df.join(places_df, how="inner", on="place_id")
    return [ner_with_places, ner_tweets_no_geo_df]


# Compara las entidades extraídas con la localización de twitter
def compare_ner_2_geo(tweet_id, place_name, full_name):
    target = 0
    # Tremenda terrajeada, pero a veces lo pone en inglés o español, con esto se resuelve creo
    if full_name.find("España") != -1:
        full_name += " Spain"
    elif full_name.find("Spain") != -1:
        full_name += " España"
    entities_recognized_list = place_name.split(", ")
    if len(entities_recognized_list) > 1:
        for entity in entities_recognized_list:
            if full_name.find(entity) != -1:
                target = 1
    else:
        if full_name.find(place_name) != -1:
            target = 1
    return [tweet_id, target]


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
