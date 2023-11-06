from ray import tune

# Training Parameters
NUM_CLASSES = 1
NUM_EPOCHS = 30
NUM_WORKERS = 9
# Logistic Regression
LR_LEARNING_RATE = 0.007399749912176894
LR_BATCH_SIZE = 32
LR_SEED = 42
# Regression
R_LEARNING_RATE = 0.001
R_BATCH_SIZE = 64
R_SEED = 42

# Dataset Size Parameters
# NB! The train, validation and test sizes must add up to the input size exactly.
INPUT_SIZE = 26000
TRAIN_SIZE = 24000
VALIDATION_SIZE = 1000
TEST_SIZE = 1000

# Not deleting above - percentage instead
TRAIN_PERCENTAGE = 0.85
VALIDATION_PERCENTAGE = 0.15
TEST_PERCENTAGE = 0.00

# Data module parameters
CSV_FILE = "/Users/christian/Desktop/Personal/University/COS711/Assignment3/data/content/Train.csv"
ROOT_DIR = (
    "/Users/christian/Desktop/Personal/University/COS711/Assignment3/data/content/train"
)

# Hyperparameter tuning
LR_DEFAULT_CONFIG = {
    "learning_rate": LR_LEARNING_RATE,
    "batch_size": LR_BATCH_SIZE,
    "optimizer": "adam",
}
R_DEFAULT_CONFIG = {
    "learning_rate": R_LEARNING_RATE,
    "batch_size": R_BATCH_SIZE,
    "optimizer": "adam",
}
SEARCH_SPACE = {
    # "learning_rate": tune.loguniform(1e-4, 1e-1),
    "learning_rate": tune.choice([0.001]),
    "batch_size": tune.choice([64]),
    "optimizer": tune.choice(["adam", "sgd", "adagrad"]),
}
TUNING_NUM_EPOCHS = 20
TUNING_NUM_SAMPLES = 5
