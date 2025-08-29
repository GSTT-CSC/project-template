def impute_data(df, categorical_columns, numerical_columns):
    for cols in categorical_columns:
        if cols in df.columns:
            df[cols] = df[cols].fillna(df[cols].mode()[0])
    for cols in numerical_columns:
        if cols in df.columns:
            df[cols] = df[cols].fillna(df[cols].mean())
    return df

def remove_outliers(df, numerical_columns, threshold=1.5):
    for col in numerical_columns:
        if col in df.columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
    return df

def normalize_data(df, numerical_columns):
    for col in numerical_columns:
        if col in df.columns:
            df[col] = (df[col] - df[col].mean()) / df[col].std()
    return df

