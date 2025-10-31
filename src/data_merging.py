import pandas as pd

def load_data(train_transaction_path, train_identity_path, test_transaction_path, test_identity_path):
    """
    Load and merge the transaction and identity datasets for both train and test.
    Returns merged train_df and test_df.
    """

    # Load datasets
    train_transaction = pd.read_csv(train_transaction_path)
    train_identity = pd.read_csv(train_identity_path)
    test_transaction = pd.read_csv(test_transaction_path)
    test_identity = pd.read_csv(test_identity_path)

    #print(f"Train transaction shape: {train_transaction.shape}")
    #print(f"Train identity shape: {train_identity.shape}")
    #print(f"Test transaction shape: {test_transaction.shape}")
    #print(f"Test identity shape: {test_identity.shape}")

    # Merge on TransactionID
    train_df = train_transaction.merge(train_identity, on='TransactionID', how='left')
    test_df = test_transaction.merge(test_identity, on='TransactionID', how='left')

    return train_df, test_df
