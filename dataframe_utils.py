from ner_utils import predicted_with_id


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
