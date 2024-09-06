import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

def load_and_preprocess_data(file_path, num=2048, samples=None):
    # Load the CSV file
    df = pd.read_csv(file_path)
    if samples:
        df = df.head(samples)

    # Extract features and convert to real
    features = df[['Poling Region Length (mm)', 'Poling Period Mismatch (nm)', 'Pump Energy (fJ)']].values

    def to_complex(s):
        return complex(s.strip('()').replace('j', 'j').replace(' ', ''))

    input_columns = [f'Input_{i}' for i in range(num)]
    input_series = df[input_columns].map(to_complex).values

    output_columns = [f'Output_{i}' for i in range(num)]
    output_series = df[output_columns].map(to_complex).values

    # Convert complex to real and reshape
    #print(features.shape, input_series.real.shape, input_series.imag.shape, output_series.real.shape)
    # first put the features and the input series together in row wise
    X = np.hstack([features, input_series.real, input_series.imag])
    y = np.hstack([output_series.real, output_series.imag])

    print(X.shape, y.shape)
    return X, y

def train_and_evaluate_model(model, X_train, X_test, y_train, y_test, model_name):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"{model_name} - MSE: {mse:.4f}, R2: {r2:.4f}")
    return y_pred

def plot_results(y_test, y_pred, model_name):
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title(f'{model_name} - Actual vs Predicted')
    plt.tight_layout()
    plt.savefig(f'{model_name}_results.png')
    plt.close()

def main():
    file_path = "/raid/robert/em/SHG_output_final.csv"
    X, y = load_and_preprocess_data(file_path, samples=10)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    #scaler = StandardScaler()
    #X_train_scaled = scaler.fit_transform(X_train)
    #X_test_scaled = scaler.transform(X_test)

    models = {
        "Neural Network": MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=1000),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
        "SVR": SVR(kernel='rbf')
    }

    for name, model in models.items():
        y_pred = train_and_evaluate_model(model, X_train, X_test, y_train, y_test, name)
        plot_results(y_test, y_pred, name)

if __name__ == "__main__":
    main()