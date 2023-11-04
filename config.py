from ray import tune

# Training Parameters
NUM_CLASSES = 1
LEARNING_RATE = 0.001
BATCH_SIZE = 64
NUM_EPOCHS = 10
NUM_WORKERS = 9

# Dataset Size Parameters
# NB! The train, validation and test sizes must add up to the input size exactly.
INPUT_SIZE = 26000
TRAIN_SIZE = 24000
VALIDATION_SIZE = 1000
TEST_SIZE = 1000

# Data module parameters
CSV_FILE = "/Users/christian/Desktop/Personal/University/COS711/Assignment3/data/content/Train.csv"
ROOT_DIR = (
    "/Users/christian/Desktop/Personal/University/COS711/Assignment3/data/content/train"
)

# Hyperparameter tuning
DEFAULT_CONFIG = {"learning_rate": LEARNING_RATE, "batch_size": BATCH_SIZE}
SEARCH_SPACE = {
    "learning_rate": tune.loguniform(1e-4, 1e-1),
    "batch_size": tune.choice([32, 64]),
}
TUNING_NUM_EPOCHS = 10
TUNING_NUM_SAMPLES = 10
