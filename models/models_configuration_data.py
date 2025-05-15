
import sys
import os

models_data_center = [
    {
        "identifier": "australia_economics",
        "country":"australia",
        "topic":"economics",
        "country_trend_data_path": "/models_data/Economy/extended_data/australia.csv",
        "regressors": [],
    },
    {
        "identifier": "australia_entertainment",
        "country":"australia",
        "topic":"entertainment",
        "country_trend_data_path": "/models_data/Entertainment/extended_data/australia.csv",
        "regressors": [
            {
                "label": "film",
                "path": "/models_data/Entertainment/extended_data/australia_trend_data/regressors/film.csv",
            },
            {
                "label": "game",
                "path": "/models_data/Entertainment/extended_data/australia_trend_data/regressors/game.csv",
            },
            {
                "label": "instagram",
                "path": "/models_data/Entertainment/extended_data/australia_trend_data/regressors/instagram.csv",
            },
            {
                "label": "song",
                "path": "/models_data/Entertainment/extended_data/australia_trend_data/regressors/song.csv",
            },
            {
                "label": "soccer",
                "path": "/models_data/Entertainment/extended_data/australia_trend_data/regressors/soccer.csv",
            },
        ],
    },
    {
        "identifier": "australia_intellectualism",
        "country":"australia",
        "topic":"intellectualism",
        "country_trend_data_path": "/models_data/Intellectualism/extended_data/australia.csv",
        "regressors": [],
    },
    {
        "identifier": "australia_nutrition",
        "country":"australia",
        "topic":"nutrition",
        "country_trend_data_path": "/models_data/Lifestyle/extended_data/australia.csv",
        "regressors": [],
    },
    {
        "identifier": "australia_politics",
        "country":"australia",
        "topic":"politics",
        "country_trend_data_path": "/models_data/Geopolitical/australia.csv",
        "regressors": [],
    },
    {
        "identifier": "canada_economics",
        "country":"canada",
        "topic":"economics",
        "country_trend_data_path": "/models_data/Economy/extended_data/canada.csv",
        "regressors": [],
    },
    {
        "identifier": "canada_entertainment",
        "country":"canada",
        "topic":"entertainment",
        "country_trend_data_path": "/models_data/Entertainment/extended_data/canada.csv",
        "regressors": [
            {
                "label": "film",
                "path": "/models_data/Entertainment/extended_data/canada_trend_data/regressors/film.csv",
            },
            {
                "label": "film",
                "path": "/models_data/Entertainment/extended_data/canada_trend_data/regressors/game.csv",
            },
            {
                "label": "film",
                "path": "/models_data/Entertainment/extended_data/canada_trend_data/regressors/soccer.csv",
            },
        ],
    },
    {
        "identifier": "canada_intellectualism",
        "country":"canada",
        "topic":"intellectualism",
        "country_trend_data_path": "/models_data/Intellectualism/extended_data/canada.csv",
        "regressors": [],
    },
    {
        "identifier": "canada_nutrition",
        "country":"canada",
        "topic":"nutrition",
        "country_trend_data_path": "/models_data/Lifestyle/extended_data/canada.csv",
        "regressors": [],
    },
    {
        "identifier": "canada_politics",
        "country":"canada",
        "topic":"politics",
        "country_trend_data_path": "/models_data/Geopolitical/canada.csv",
        "regressors": [],
    },
    {
        "identifier": "canada_economics",
        "country":"canada",
        "topic":"economics",
        "country_trend_data_path": "/models_data/Economy/extended_data/usa.csv",
        "regressors": [
            {
                "label": "diplomacy",
                "path": "/models_data/Geopolitical/extended_data/usa_trend_data/regressors/diplomacy.csv",
            },
            {
                "label": "economy",
                "path": "/models_data/Geopolitical/extended_data/usa_trend_data/regressors/economy.csv",
            },
            {
                "label": "elections",
                "path": "/models_data/Geopolitical/extended_data/usa_trend_data/regressors/elections.csv",
            },
            {
                "label": "war",
                "path": "/models_data/Geopolitical/extended_data/usa_trend_data/regressors/war.csv",
            },
            {
                "label": "immigration",
                "path": "/models_data/Geopolitical/extended_data/usa_trend_data/regressors/immigration.csv",
            },
        ],
    },
    {
        "identifier": "usa_entertainment",
        "country":"usa",
        "topic":"entertainment",
        "country_trend_data_path": "/models_data/Entertainment/extended_data/usa.csv",
        "regressors": [],
    },
    {
        "identifier": "usa_intellectualism",
        "country":"usa",
        "topic":"intellectualism",
        "country_trend_data_path": "/models_data/Intellectualism/extended_data/usa.csv",
        "regressors": [
            # {
            #     "label": "book",
            #     "path": "/models_data/Intellectualism/extended_data/usa_trend_data/regressors/book.csv",
            # },
            # {
            #     "label": "education",
            #     "path": "/models_data/Intellectualism/extended_data/usa_trend_data/regressors/education.csv",
            # },
            # {
            #     "label": "history",
            #     "path": "/models_data/Intellectualism/extended_data/usa_trend_data/regressors/history.csv",
            # },
            # {
            #     "label": "literature",
            #     "path": "/models_data/Intellectualism/extended_data/usa_trend_data/regressors/literature.csv",
            # },
            # {
            #     "label": "religion",
            #     "path": "/models_data/Intellectualism/extended_data/usa_trend_data/regressors/religion.csv",
            # },
            #     {
            #     "label": "novel",
            #     "path": "/models_data/Intellectualism/extended_data/usa_trend_data/regressors/novel.csv",
            # },
            {
                "label": "culture",
                "path": "/models_data/Intellectualism/extended_data/usa_trend_data/regressors/culture.csv",
            },
            {
                "label": "nation",
                "path": "/models_data/Intellectualism/extended_data/usa_trend_data/regressors/nation.csv",
            },
            {
                "label": "america",
                "path": "/models_data/Intellectualism/extended_data/usa_trend_data/regressors/america.csv",
            },
        ],
    },
    {
        "identifier": "usa_nutrition",
        "country":"usa",
        "topic":"nutrition",
        "country_trend_data_path": "/models_data/Lifestyle/extended_data/usa.csv",
        "regressors": [],
    },
    {
        "identifier": "usa_politics",
        "country":"usa",
        "topic":"politics",
        "country_trend_data_path": "/models_data/Geopolitical/usa.csv",
        "regressors": [
            {
                "label": "diplomacy",
                "path": "/models_data/Geopolitical/extended_data/usa_trend_data/regressors/diplomacy.csv",
            },
            {
                "label": "economy",
                "path": "/models_data/Geopolitical/extended_data/usa_trend_data/regressors/economy.csv",
            },
            {
                "label": "elections",
                "path": "/models_data/Geopolitical/extended_data/usa_trend_data/regressors/elections.csv",
            },
            {
                "label": "war",
                "path": "/models_data/Geopolitical/extended_data/usa_trend_data/regressors/war.csv",
            },
            {
                "label": "immigration",
                "path": "/models_data/Geopolitical/extended_data/usa_trend_data/regressors/immigration.csv",
            },
        ],
    },
]
