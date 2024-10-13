from sklearn.model_selection import train_test_split
import numpy as np

if __name__ == "__main__":

    # load all dataset
    labels = np.load("labels.npy")
    codemaps = np.load("codemaps.npy")
    # vqs = np.load("vqs.npy")

    print("Distribution of labels:")
    print(np.sum(labels, axis=0))

    # split data into training, validation and test datasets

    X_train, X_test, X_vqs_train, X_vqs_test, y_train, y_test = train_test_split(
        codemaps, vqs, labels, test_size=0.15, random_state=0
    )

    X_train, X_val, X_vqs_train, X_vqs_val, y_train, y_val = train_test_split(
        X_train, X_vqs_train, y_train, test_size=0.20, random_state=0
    )

    # Calculate the number of samples in each set

    total_samples = len(codemaps)
    train_samples = len(X_train)
    val_samples = len(X_val)
    test_samples = len(X_test)

    # Calculate the percentage of samples in each set
    train_percentage = (train_samples / total_samples) * 100
    val_percentage = (val_samples / total_samples) * 100
    test_percentage = (test_samples / total_samples) * 100

    print(f"\nTotal samples: {total_samples}")
    print(f"- Training: {train_percentage:.1f}%, {train_samples} samples")
    print(f"- Validation: {val_percentage:.1f}%, {val_samples} samples")
    print(f"- Testing: {test_percentage:.1f}%, {test_samples} samples")

    np.save("X_train.npy", X_train)
    # np.save("vqs_train.npy", X_vqs_train)
    np.save("y_train.npy", y_train)
    np.save("X_val.npy", X_val)
    # np.save("vqs_val.npy", X_vqs_val)
    np.save("y_val.npy", y_val)
    np.save("X_test.npy", X_test)
    # np.save("vqs_test.npy", X_vqs_test)
    np.save("y_test.npy", y_test)
