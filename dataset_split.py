import pandas as pd
from sklearn.model_selection import train_test_split


#Funtion splits the data in Train, Validation and Test
def split_data(df, oversampled=False):
    X = df[["reviews_pipeline_0", "reviews_pipeline_1", "class"]]
    y = df["class"]
    X_train_valid, X_test, y_train_valid, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    X_train, X_valid, y_train, y_valid = train_test_split(X_train_valid, y_train_valid, test_size = 0.3, random_state = 42, stratify=y_train_valid)

    if oversampled:
        X_train = perform_oversample(X_train)
        
    y_train =  X_train["class"]    
    X_train = X_train[["reviews_pipeline_0", "reviews_pipeline_1"]]

    y_train = y_train.to_numpy()
    y_test = y_test.to_numpy()

    return X_train, X_valid, X_test, y_train, y_valid, y_test


#Function performs that performs oversample
def perform_oversample(df):

    class_1,class_0 = df["class"].value_counts()
    c0 = df[df['class'] == 0]
    c1 = df[df['class'] == 1]

    df_0 = c0.sample(round(class_1/3), replace=True)
    oversampled_df = pd.concat([c1,df_0], axis=0)

    return oversampled_df


#