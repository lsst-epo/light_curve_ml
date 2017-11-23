import os

import numpy as np
import upsilon


def loadMacho():
    # TODO: loop files in macho dir
    fileDir = os.path.join(os.environ.get("LSST"), "data/macho", "F_1.3319")
    trainData = np.genfromtxt(fileDir, delimiter=';', dtype=None)

    mjds = []
    redMags = []
    redErrors = []
    blueMags = []
    blueErrors = []
    for line in trainData:
        mjds.append(line[4])
        redMags.append(float(line[9]))
        redErrors.append(float(line[10]))
        blueMags.append(float(line[24]))
        blueErrors.append(float(line[25]))

    return [[mjds, redMags, redErrors], [mjds, blueMags, blueErrors]]


def main():
    # Loads baked classification model
    rf_model = upsilon.load_rf_model()
    set_of_light_curves = loadMacho()
    for datetimes, magnitudes, errors in set_of_light_curves:
        # Extract features from each light curve and predict its class.
        date = np.array(datetimes)
        mag = np.array(magnitudes)
        err = np.array(errors)

        # Extract features
        e_features = upsilon.ExtractFeatures(date, mag, err)
        e_features.run()
        features = e_features.get_features()

        # Classify the light curve
        label, probability, flag = upsilon.predict(rf_model, features)
        print([label, probability, flag])


if __name__ == "__main__":
    main()
