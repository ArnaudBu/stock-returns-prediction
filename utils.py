from pytorch_tabnet.tab_model import TabNetClassifier
import torch
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler


def process_data(df, date_valid, date_test, features, target):
    data = df.copy()
    # split datasets
    if "set" not in data.columns:
        data["set"] = "train"
        data.loc[data.date > date_valid, "set"] = "valid"
        data.loc[data.date > date_test, "set"] = "test"
    train_indices = data[data.set == "train"].index
    valid_indices = data[data.set == "valid"].index
    test_indices = data[data.set == "test"].index
    indices = data.set.values

    # Select data
    data = data[features + [target]]

    # Get categorical features and preprocess
    nunique = data.nunique()
    types = data.dtypes
    categorical_columns = []
    categorical_dims = {}
    for col in data.columns:
        if types[col] == 'object' or nunique[col] < 200:
            l_enc = LabelEncoder()
            data[col] = data[col].fillna("Unknown")
            data[col] = l_enc.fit_transform(data[col].values)
            categorical_columns.append(col)
            categorical_dims[col] = len(l_enc.classes_)
        else:
            data[col].fillna(data[col].mean(), inplace=True)
            scaler = StandardScaler()
            data[[col]] = scaler.fit_transform(data[[col]])
    cat_idxs = [i for i, f in enumerate(features) if f in categorical_columns]
    cat_dims = [categorical_dims[f] for i, f in enumerate(features) if f in categorical_columns]

    # Define model
    clf = TabNetClassifier(cat_idxs=cat_idxs,
                           cat_dims=cat_dims,
                           cat_emb_dim=1,
                           optimizer_fn=torch.optim.Adam,
                           optimizer_params=dict(lr=1e-2),
                           scheduler_params={"step_size": 10,
                                             "gamma": 0.9},
                           scheduler_fn=torch.optim.lr_scheduler.StepLR,
                           mask_type='entmax'
                           )

    # Datasets
    X_train = data[features].values[train_indices]
    y_train = data[target].values[train_indices]
    X_valid = data[features].values[valid_indices]
    y_valid = data[target].values[valid_indices]
    X_test = data[features].values[test_indices]
    y_test = data[target].values[test_indices]

    return clf, X_train, y_train, X_valid, y_valid, X_test, y_test, indices
