import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def preprocess(path='data/raw/california.csv', output_dir='data/processed/'):
    df = pd.read_csv(path)
    X = df.drop('MedHouseVal', axis=1)
    y = df['MedHouseVal']
                    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    pd.DataFrame(X_train).to_csv(f'{output_dir}X_train.csv', index=False)
    pd.DataFrame(X_test).to_csv(f'{output_dir}X_test.csv', index=False)
    y_train.to_csv(f'{output_dir}y_train.csv', index=False)
    y_test.to_csv(f'{output_dir}y_test.csv', index=False)

if __name__ == "__main__":
    preprocess()

