import pandas as pd
from sklearn.model_selection import train_test_split

def load_accepted_data(filepath):
    """
    Load and preprocess the accepted loans dataset.
    """
    df = pd.read_csv(filepath, low_memory=False)

    # Drop unnecessary columns
    drop_cols = ['id', 'member_id', 'emp_title', 'title', 'zip_code']
    df.drop(columns=drop_cols, errors='ignore', inplace=True)

    # Encode target variable safely
    df['loan_status'] = df['loan_status'].apply(lambda x: 1 if isinstance(x, str) and 'Charged Off' in x else 0)

    # Handle missing values
    df.fillna(0, inplace=True)

    # Encode categorical features
    cat_cols = df.select_dtypes(include=['object']).columns
    df[cat_cols] = df[cat_cols].astype('category').apply(lambda x: x.cat.codes)

    return df

def load_rejected_data(filepath):
    """
    Load and preprocess the rejected loans dataset.
    """
    df = pd.read_csv(filepath, low_memory=False)

    # Drop unnecessary columns
    drop_cols = ['Amount Requested', 'Application Date']
    df.drop(columns=drop_cols, errors='ignore', inplace=True)

    # Rename columns to match accepted dataset structure
    rename_cols = {
        'Risk_Score': 'fico_score',
        'Debt-To-Income Ratio': 'dti',
        'State': 'addr_state',
        'Zip Code': 'zip_code'
    }
    df.rename(columns=rename_cols, inplace=True)

    # Handle missing values
    df.fillna(0, inplace=True)

    # Encode categorical features
    cat_cols = df.select_dtypes(include=['object']).columns
    df[cat_cols] = df[cat_cols].astype('category').apply(lambda x: x.cat.codes)

    return df

def split_data(df):
    """
    Split accepted loans into train-test sets.
    """
    X = df.drop(columns=['loan_status'])
    y = df['loan_status']
    return train_test_split(X, y, test_size=0.2, random_state=42)

if __name__ == "__main__":
    df_accepted = load_accepted_data("data/accepted_2007_to_2018Q4.csv")
    df_rejected = load_rejected_data("data/rejected_2007_to_2018Q4.csv")

    print(f"Accepted Loans Shape: {df_accepted.shape}")
    print(f"Rejected Loans Shape: {df_rejected.shape}")

    X_train, X_test, y_train, y_test = split_data(df_accepted)
    print(f"Training Data: {X_train.shape}, Test Data: {X_test.shape}")
