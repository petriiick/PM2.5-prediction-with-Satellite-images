import pandas as pd
import pickle


def get_distance(n_df, loc, path):
    for image in n_df:
        for index, location in loc.items():
            image["distance"].append(
                {
                    index: np.sqrt(
                        (image["lat"] - location["lat"]) ** 2
                        + (image["lat"] - location["lat"]) ** 2
                    )
                }
            )
    pickle.dump(n_df, open(path, "wb"))


def main():
    n_df = pd.read_pickle("/work/ld243/inter_label.pkl")
    loc = pd.read_pickle("/work/ld243/sensor_locations.pkl")
    path = "/work/ld243/label_fin.pkl"

    get_distance(n_df, loc, path)


main()
