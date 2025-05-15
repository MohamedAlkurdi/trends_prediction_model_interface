import pandas as pd

cosine_similarity_values = {
    "AFRICA": {
        "Economy": pd.DataFrame(
            {
                "Kenya": ["-", 99.37, 98.1],
                "Nigeria": [99.37, "-", 97.47],
                "SouthAfrica": [98.1, 97.47, "-"],
            },
            index=["Kenya", "Nigeria", "SouthAfrica"],
        ),
        "Technology and Science": pd.DataFrame(
            {
                "Kenya": ["-", 80.38, 81.01],
                "Nigeria": [80.38, "-", 79.75],
                "SouthAfrica": [81.01, 79.75, "-"],
            },
            index=["Kenya", "Nigeria", "SouthAfrica"],
        ),
        "Entertainment": pd.DataFrame(
            {
                "Kenya": ["-", 60.13, 60.76],
                "Nigeria": [60.13, "-", 59.49],
                "SouthAfrica": [60.76, 59.49, "-"],
            },
            index=["Kenya", "Nigeria", "SouthAfrica"],
        ),
        "Lifestyle": pd.DataFrame(
            {
                "Kenya": ["-", 70.89, 58.23],
                "Nigeria": [70.89, "-", 57.59],
                "SouthAfrica": [58.23, 57.59, "-"],
            },
            index=["Kenya", "Nigeria", "SouthAfrica"],
        ),
        "Accident": pd.DataFrame(
            {
                "Kenya": ["-", 55.7, 48.1],
                "Nigeria": [55.7, "-", 49.37],
                "SouthAfrica": [48.1, 49.37, "-"],
            },
            index=["Kenya", "Nigeria", "SouthAfrica"],
        ),
        "Geopolitical": pd.DataFrame(
            {
                "Kenya": ["-", 72.78, 63.92],
                "Nigeria": [72.78, "-", 62.03],
                "SouthAfrica": [63.92, 62.03, "-"],
            },
            index=["Kenya", "Nigeria", "SouthAfrica"],
        ),
        "Intellectualism": pd.DataFrame(
            {
                "Kenya": ["-", 65.82, 57.59],
                "Nigeria": [65.82, "-", 57.59],
                "SouthAfrica": [57.59, 57.59, "-"],
            },
            index=["Kenya", "Nigeria", "SouthAfrica"],
        ),
    },
    "EUROPE": {
        "Economy": pd.DataFrame(
            {
                "Denmark": ["-", 92.41, 88.61],
                "UK": [92.41, "-", 93.67],
                "Finland": [88.61, 93.67, "-"],
            },
            index=["Denmark", "UK", "Finland"],
        ),
        "Technology and Science": pd.DataFrame(
            {
                "Denmark": ["-", 53.8, 56.96],
                "UK": [53.8, "-", 50.63],
                "Finland": [56.96, 50.63, "-"],
            },
            index=["Denmark", "UK", "Finland"],
        ),
        "Entertainment": pd.DataFrame(
            {
                "Denmark": ["-", 77.85, 86.71],
                "UK": [77.85, "-", 73.42],
                "Finland": [86.71, 73.42, "-"],
            },
            index=["Denmark", "UK", "Finland"],
        ),
        "Lifestyle": pd.DataFrame(
            {
                "Denmark": ["-", 50.0, 48.1],
                "UK": [50.0, "-", 46.2],
                "Finland": [48.1, 46.2, "-"],
            },
            index=["Denmark", "UK", "Finland"],
        ),
        "Accident": pd.DataFrame(
            {
                "Denmark": ["-", 51.9, 49.37],
                "UK": [51.9, "-", 60.76],
                "Finland": [49.37, 60.76, "-"],
            },
            index=["Denmark", "UK", "Finland"],
        ),
        "Geopolitical": pd.DataFrame(
            {
                "Denmark": ["-", 48.73, 44.94],
                "UK": [48.73, "-", 43.67],
                "Finland": [44.94, 43.67, "-"],
            },
            index=["Denmark", "UK", "Finland"],
        ),
        "Intellectualism": pd.DataFrame(
            {
                "Denmark": ["-", 50.0, 44.3],
                "UK": [50.0, "-", 46.2],
                "Finland": [44.3, 46.2, "-"],
            },
            index=["Denmark", "UK", "Finland"],
        ),
    },
    "NORTH_AMERICA_AUSTRALIA": {
        "Economy": pd.DataFrame(
            {
                "Australia": ["-", 94.94, 96.84],
                "Canada": [94.94, "-", 95.57],
                "USA": [96.84, 95.57, "-"],
            },
            index=["Australia", "Canada", "USA"],
        ),
        "Technology and Science": pd.DataFrame(
            {
                "Australia": ["-", 50.63, 55.7],
                "Canada": [50.63, "-", 59.49],
                "USA": [55.7, 59.49, "-"],
            },
            index=["Australia", "Canada", "USA"],
        ),
        "Entertainment": pd.DataFrame(
            {
                "Australia": ["-", 91.14, 85.44],
                "Canada": [91.14, "-", 91.77],
                "USA": [85.44, 91.77, "-"],
            },
            index=["Australia", "Canada", "USA"],
        ),
        "Lifestyle": pd.DataFrame(
            {
                "Australia": ["-", 68.35, 72.15],
                "Canada": [68.35, "-", 65.19],
                "USA": [72.15, 65.19, "-"],
            },
            index=["Australia", "Canada", "USA"],
        ),
        "Accident": pd.DataFrame(
            {
                "Australia": ["-", 74.05, 79.75],
                "Canada": [74.05, "-", 74.68],
                "USA": [79.75, 74.68, "-"],
            },
            index=["Australia", "Canada", "USA"],
        ),
        "Geopolitical": pd.DataFrame(
            {
                "Australia": ["-", 46.84, 47.47],
                "Canada": [46.84, "-", 62.03],
                "USA": [47.47, 62.03, "-"],
            },
            index=["Australia", "Canada", "USA"],
        ),
        "Intellectualism": pd.DataFrame(
            {
                "Australia": ["-", 58.23, 70.25],
                "Canada": [58.23, "-", 55.7],
                "USA": [70.25, 55.7, "-"],
            },
            index=["Australia", "Canada", "USA"],
        ),
    },
    "WEST_ASIA": {
        "Economy": pd.DataFrame(
            {
                "Australia": ["-", 95.57, 93.67],
                "Canada": [95.57, "-", 94.3],
                "USA": [93.67, 94.3, "-"],
            },
            index=["Australia", "Canada", "USA"],
        ),
        "Technology and Science": pd.DataFrame(
            {
                "Australia": ["-", 67.72, 60.76],
                "Canada": [67.72, "-", 63.29],
                "USA": [60.76, 63.29, "-"],
            },
            index=["Australia", "Canada", "USA"],
        ),
        "Entertainment": pd.DataFrame(
            {
                "Australia": ["-", 78.48, 81.65],
                "Canada": [78.48, "-", 91.77],
                "USA": [81.65, 91.77, "-"],
            },
            index=["Australia", "Canada", "USA"],
        ),
        "Lifestyle": pd.DataFrame(
            {
                "Australia": ["-", 62.66, 52.53],
                "Canada": [62.66, "-", 44.3],
                "USA": [52.53, 44.3, "-"],
            },
            index=["Australia", "Canada", "USA"],
        ),
        "Accident": pd.DataFrame(
            {
                "Australia": ["-", 52.53, 49.37],
                "Canada": [52.53, "-", 47.47],
                "USA": [49.37, 47.47, "-"],
            },
            index=["Australia", "Canada", "USA"],
        ),
        "Geopolitical": pd.DataFrame(
            {
                "Australia": ["-", 76.58, 55.06],
                "Canada": [76.58, "-", 50.63],
                "USA": [55.06, 50.63, "-"],
            },
            index=["Australia", "Canada", "USA"],
        ),
        "Intellectualism": pd.DataFrame(
            {
                "Australia": ["-", 58.86, 66.46],
                "Canada": [58.86, "-", 55.06],
                "USA": [66.46, 55.06, "-"],
            },
            index=["Australia", "Canada", "USA"],
        ),
    },
}


