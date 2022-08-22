import os
import phenoKOR as pk
import numpy as np
import pandas as pd
import seaborn as sns
import math
import matplotlib.pyplot as plt


def check_stationarity(path):
    x, *y = pk.load_csv(path)


if __name__ == "__main__":
    root = "/Users/beom/Desktop/git/data/knps/"

    check_stationarity(root + "day_8_data/2021_jiri_final.csv")