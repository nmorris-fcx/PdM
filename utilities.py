import numpy as np
import pandas as pd
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import PolynomialFeatures, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_selection import RFE
from sklearn.neighbors import LocalOutlierFactor
import plotly.express as px
from plotly.offline import plot


def line_plot(data, x, y, color=None, title=None, font_size=None, name="line plot"):
    fig = px.line(data, x=x, y=y, color=color, title=title)
    fig.update_layout(font=dict(size=font_size))
    plot(fig, filename=name + ".html")


def bar_plot(data, x, y, color=None, title=None, font_size=None, name="bar plot"):
    fig = px.bar(data, x=x, y=y, color=color, title=title)
    fig.update_layout(font=dict(size=font_size))
    plot(fig, filename=name + ".html")


def scatter_plot(data, x, y, color=None, title=None, font_size=None, name="line plot"):
    fig = px.scatter(data, x=x, y=y, color=color, title=title)
    fig.update_layout(font=dict(size=font_size))
    plot(fig, filename=name + ".html")

 def string_binaries(data):
    """
    Converts any string variables into binary variables
    """
    # determine which columns are strings
    _columns = data.columns
    _dtypes = data.dtypes
    _str = _columns[np.where(_dtypes == "object")[0]]
    if len(_str) > 0:
        # convert any string columns to binary columns
        data = pd.get_dummies(data, columns=_str)
    return data


def variance_threshold(data, threshold=0):
    """
    Removes constant columns or low variance columns
    """
    selector = VarianceThreshold(threshold)
    selector.fit(data)
    return data[data.columns[selector.get_support(indices=True)]]


def polynomials(data, degree=2):
    """
    Engineers polynomial features up to a degree
    """
    poly = PolynomialFeatures(degree, include_bias=False)
    _columns = data.columns
    data = pd.DataFrame(poly.fit_transform(data))
    data.columns = poly.get_feature_names(_columns)
    return data


def rf_rfe(X, Y, classifier=False, select=0.5, step=0.05, n_jobs=1):
    """
    Use a Random Forest to do Recursive Feature Elimination
    This selects the most useful features for modeling
    """
    if classifier:
        selector = RFE(
            RandomForestClassifier(
                n_estimators=25,
                max_depth=10,
                min_samples_leaf=1,
                max_features="sqrt",
                random_state=42,
                class_weight="balanced_subsample",
                n_jobs=n_jobs,
            ),
            step=step,
            n_features_to_select=select,
            verbose=1,
        )
    else:
        selector = RFE(
            RandomForestRegressor(
                n_estimators=25,
                max_depth=10,
                min_samples_leaf=1,
                max_features="sqrt",
                random_state=42,
                n_jobs=n_jobs,
            ),
            step=step,
            n_features_to_select=select,
            verbose=1,
        )
    selector.fit(X, Y)
    return X[X.columns[selector.get_support(indices=True)]]


def min_max_scale(data):
    """
    Scale columns to take on a min value of 0 and a max value of 1:
        scale_x = (max - x) / (max - min)
    """
    _columns = data.columns
    scaler = MinMaxScaler()
    data = scaler.fit_transform(data)
    data = pd.DataFrame(data, columns=_columns)
    return data


def find_inliers(data, percent=0.1):
    """
    Locates outliers in the data using k-nearest neighbors
    Returns the row indices of inliers
    """
    model = LocalOutlierFactor(n_neighbors=20, leaf_size=30, novelty=False, n_jobs=1)
    model.fit(min_max_scale(data))
    cutoff = np.quantile(model.negative_outlier_factor_, percent)
    good_idx = np.where(model.negative_outlier_factor_ > cutoff)[0]
    return good_idx
