import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

def preprocess_data(train_df, test_df):
    """
    Cleans the data, handles missing values, encodes categorical features,
    creates time-based features, drops irrelevant columns, and aligns test features to train.
    """
      
    #train_df = train_df.copy()
    #test_df = test_df.copy()
    
    # Define numeric columns to keep
    base_numeric = [
            'TransactionID', 'isFraud', 'TransactionDT', 'TransactionAmt',
            'card1', 'card2', 'card3', 'card5', 'addr1', 'addr2', 'dist1', 'dist2'
        ]

    high_corr_numeric = [
            'V257', 'V246', 'V244', 'V242', 'V201', 'V200', 'V189', 'V188', 'V258',
            'V45', 'V228', 'V44', 'V86', 'V87', 'V170', 'V52', 'V230', 'V199', 'V51',
            'V171', 'V40', 'V243', 'V190', 'V39', 'C2', 'C8', 'C12', 'C1', 'C4',
            'C10', 'C7', 'C11', 'C6', 'C14', 'C3', 'D14', 'C13', 'D12', 'C5', 'C9',
            'D9', 'D11', 'D3', 'D6', 'D13', 'D5', 'D1', 'D4', 'D10', 'D15', 'D2',
            'D7', 'D8'
        ]
    
    #categorical features 
    cate_features = ['M4', 'M3', 'M2', 'M1', 'M6', 'P_emaildomain', 'card4', 'card6', 'ProductCD']

    keep_numeric_cols = base_numeric + high_corr_numeric

    # Identify numeric columns in both datasets
    numeric_cols_train = train_df.select_dtypes(include=['int64', 'float64']).columns
    numeric_cols_test = test_df.select_dtypes(include=['int64', 'float64']).columns

    # Drop only unwanted numeric columns (keep categoricals intact)
    drop_num_train = [col for col in numeric_cols_train if col not in keep_numeric_cols]
    drop_num_test = [col for col in numeric_cols_test if col not in keep_numeric_cols]

    train_df = train_df.drop(columns=drop_num_train, errors='ignore')
    test_df = test_df.drop(columns=drop_num_test, errors='ignore')

    # Fill missing numeric values
    num_cols_to_fill = [col for col in keep_numeric_cols if col in train_df.columns and col != 'isFraud']
    train_df[num_cols_to_fill] = train_df[num_cols_to_fill].fillna(-999)
    test_df[num_cols_to_fill] = test_df[num_cols_to_fill].fillna(-999)

    # Generate time-based features 
    for df in [train_df, test_df]:
        if 'TransactionDT' in df.columns:
            df['Transaction_days'] = df['TransactionDT'] // (24 * 3600)
            df['Transaction_hours'] = (df['TransactionDT'] // 3600) % 24
            df['Transaction_weekday'] = df['Transaction_days'] % 7
            df.drop(columns=['TransactionDT'], inplace=True, errors='ignore')

    # Drop TransactionID 
    train_df = train_df.drop(columns=['TransactionID'], errors='ignore')
    test_df = test_df.drop(columns=['TransactionID'], errors='ignore')


    # Fill missing values with 'missing'
    for col in cate_features:
        if col in train_df.columns:
            train_df[col] = train_df[col].fillna('missing')
        if col in test_df.columns:
            test_df[col] = test_df[col].fillna('missing')

    # Frequency Encoding for P_emaildomain 
    if 'P_emaildomain' in train_df.columns:
        freq = train_df['P_emaildomain'].value_counts(normalize=True)
        train_df['P_emaildomain'] = train_df['P_emaildomain'].map(freq)
        test_df['P_emaildomain'] = test_df['P_emaildomain'].map(freq).fillna(0)

    # Aggregate behaviour per card1 
    card1_agg = train_df.groupby('card1').agg({
        'TransactionAmt': ['mean', 'std', 'count'],
        'isFraud': ['mean']
    }).reset_index()

    card1_agg.columns = ['card1', 'card1_amt_mean', 'card1_amt_std', 'card1_txn_count', 'card1_fraud_rate']

    # Merge back to train and test
    train_df= train_df.merge(card1_agg, on='card1', how='left')
    test_df = test_df.merge(card1_agg, on='card1', how='left')

    # Drop the raw card1
    train_df.drop('card1', axis=1, inplace=True)
    test_df.drop('card1', axis=1, inplace=True)

    train_df = pd.get_dummies(train_df, columns=['card4', 'card6'], drop_first=True)

    col_drop = ['card4_mastercard', 'card4_missing', 'card4_visa','card6_debit','card6_debit or credit', 'card6_missing']
    train_df = train_df.drop(columns=col_drop)

    # Label Encoding for Other Categorical Columns ---
    for col in [c for c in cate_features if c != 'P_emaildomain']:
        if col in train_df.columns:
            le = LabelEncoder()
            le.fit(list(train_df[col].astype(str)) + list(test_df[col].astype(str)))  # fit on both to avoid unseen classes
            train_df[col] = le.transform(train_df[col].astype(str))
            test_df[col] = le.transform(test_df[col].astype(str))

    # Drop Irrelevant Categorical Columns 
    all_cat_features = [
            'ProductCD', 'card4', 'card6', 'P_emaildomain', 'R_emaildomain',
            'M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'M8', 'M9',
            'id_12', 'id_15', 'id_16', 'id_23', 'id_27', 'id_28', 'id_29',
            'id_30', 'id_31', 'id_33', 'id_34', 'id_35', 'id_36', 'id_37', 'id_38',
            'DeviceType', 'DeviceInfo'
        ]

    # columns to drop
    drop_cat_columns = [col for col in all_cat_features if col not in cate_features]
    train_df = train_df.drop(columns=[col for col in drop_cat_columns if col in train_df.columns], errors='ignore')
    test_df = test_df.drop(columns=[col for col in drop_cat_columns if col in test_df.columns], errors='ignore')

    # Align test features to train
    missing_cols = set(train_df.columns) - set(test_df.columns)
    for col in missing_cols:
        if col != 'isFraud':  # don't add target to test
            test_df[col] = -999
    test_df = test_df[train_df.drop(columns=['isFraud']).columns]  # reorder
    
    return train_df, test_df

def split_data(train_df):
    #--split data
    X = train_df.drop('isFraud', axis=1)
    y = train_df['isFraud']

    X_train, X_test, y_train,y_test = train_test_split(X,
                                                        y, 
                                                        test_size=0.2, 
                                                        random_state=42, 
                                                        stratify=y)
    return X_train, X_test, y_train, y_test