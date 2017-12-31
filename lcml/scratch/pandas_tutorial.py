import datetime

from matplotlib.pyplot import style
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pandas_datareader import data


def main():
    style.use("ggplot")

    # start = datetime.datetime(2015, 1, 1)
    # end = datetime.datetime(2015, 3, 1)
    # df = data.DataReader("XOM", "google", start, end, retry_count=10)
    #
    # print df.head()
    #
    # df.plot()
    # plt.show()

    stats = {"Day": [1, 2, 3, 4, 5, 6],
             "Visitors": [43, 53, 34, 45, 64, 34],
             "Bounce_Rate": [65, 72, 62, 64, 54, 66]}

    df = pd.DataFrame(stats)
    # print(df)
    # print(df.head(2))
    #
    # print(df.tail(2))

    # Index returning a new DataFrame
    # print(df.set_index("Day"))

    # makes changes to df in-place
    df.set_index("Day", inplace=True)

    # print(df["Bounce_Rate"])
    # print(df[["Bounce_Rate", "Visitors"]])
    # print(df.Visitors.tolist())
    print(np.array(df[["Bounce_Rate", "Visitors"]]))

    # convert df to ndarray to df
    df2 = pd.DataFrame(np.array(df[["Bounce_Rate", "Visitors"]]))
    print(df2)


if __name__ == "__main__":
    main()
