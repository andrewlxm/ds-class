import numpy as np
from sklearn.metrics import accuracy_score
from models import build_models, estimate_flops
from perf_counter import PAPICounter
import csv
import os
import pickle
import psutil
import time

def load_data():
    X_train = np.load('data/mnist1_features_train.npy', allow_pickle=True)
    y_train = np.load('data/mnist1_labels_train.npy', allow_pickle=True)
    X_test = np.load('data/mnist1_features_test.npy', allow_pickle=True)
    y_test = np.load('data/mnist1_labels_test.npy', allow_pickle=True)

    n_samples, n_features = X_train.shape
    print('Train data contains: {} samples of dimension {}'.format(n_samples, n_features))
    print('Test data contains: {} samples'.format(X_test.shape[0]))

    return X_train, y_train, X_test, y_test

def benchmark_model(model, X_train, y_train, X_test, y_test):

    process = psutil.Process(os.getpid())
    counter = PAPICounter()

    # Train
    mem_before = process.memory_info().rss
    t0 = time.perf_counter()
    counter.start()

    model.fit(X_train, y_train)

    train_flops = counter.stop()
    train_time = time.perf_counter() - t0
    mem_after = process.memory_info().rss

    # Predict
    t0 = time.perf_counter()
    counter.start()
    y_pred = model.predict(X_test)
    pred_flops = counter.stop()
    pred_time = time.perf_counter() - t0

    acc = accuracy_score(y_test, y_pred)

    # Model size
    temp_file = "temp_model.pkl"
    with open(temp_file, "wb") as f:
        pickle.dump(model, f)

    model_size = os.path.getsize(temp_file) / (1024**2)
    os.remove(temp_file)

    return {
        "train_time": train_time,
        "train_flops": train_flops,
        "predict_time": pred_time,
        "predict_flops": pred_flops,
        "accuracy": acc,
        "model_size": model_size,
    }


def run_benchmark():

    X_train, y_train, X_test, y_test = load_data()

    d = X_train.shape[1]
    n_train = X_train.shape[0]

    models = build_models()

    results = []

    for name, model in models.items():

        print(f"\n=== {name} ===")

        metrics = benchmark_model(
            model, X_train, y_train, X_test, y_test
        )

        flops_per_sample, flops_total_eval = estimate_flops(name, model, d, n_train, X_test.shape[0])

        results.append({
            "model": name,
            "estimate_flops_per_sample": flops_per_sample,
            "estimate_flops_total_eval": flops_total_eval,
            "train_time": metrics["train_time"],
            "perf_train_count": metrics["train_flops"],
            "predict_time": metrics["predict_time"],
            "perf_predict_count": metrics["predict_flops"],
            "accuracy": metrics["accuracy"],
            "model_size": metrics["model_size"],
        })

    return results
           
        
#Save results to CSV
def save_csv(results, filename="benchmark.csv"):

    with open(filename, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    print(f"Saved full benchmark to {filename}")

if __name__ == "__main__":
    results = run_benchmark()
    save_csv(results, "benchmark_perf.csv")