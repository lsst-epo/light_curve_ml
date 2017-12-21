from feets.datasets import macho
from feets.datasets.ogle3 import load_OGLE3_catalog, fetch_OGLE3
from feets.extractors.core import DATA_TIME, DATA_MAGNITUDE, DATA_ERROR


def peekMacho():
    """Only provides small number of macho's"""
    print(macho.available_MACHO_lc())


def peekOgle():
    if 1:
        df = load_OGLE3_catalog()
        # print(list(df))
        validIds = [id for id in df["ID"] if id != "-99.99"]
        print("Valid OGLE3 ids: %s" % len(validIds))
        count = 0
        for vid in validIds:
            bunch = fetch_OGLE3(vid)
            if bunch and bunch.bands:
                for subBunch in bunch.bands.values():
                    if len(subBunch[DATA_TIME]):
                        count += 1

        print("Valid rate %s / %s" % (count, len(validIds)))

    if 0:
        print(fetch_OGLE3("OGLE-BLG-LPV-228732"))


if __name__ == "__main__":
    peekOgle()