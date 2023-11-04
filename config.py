from ray import tune

# Training Parameters
NUM_CLASSES = 1
NUM_EPOCHS = 5
NUM_WORKERS = 9
# Logistic Regression
LR_LEARNING_RATE = 0.001
LR_BATCH_SIZE = 64
# Regression
R_LEARNING_RATE = 0.001
R_BATCH_SIZE = 64

# Dataset Size Parameters
# NB! The train, validation and test sizes must add up to the input size exactly.
INPUT_SIZE = 26000
TRAIN_SIZE = 24000
VALIDATION_SIZE = 1000
TEST_SIZE = 1000

# Not deleting above - percentage instead
TRAIN_PERCENTAGE = 0.9
VALIDATION_PERCENTAGE = 0.05
TEST_PERCENTAGE = 0.05

# Data module parameters
CSV_FILE = "/Users/christian/Desktop/Personal/University/COS711/Assignment3/data/content/Train.csv"
ROOT_DIR = (
    "/Users/christian/Desktop/Personal/University/COS711/Assignment3/data/content/train"
)

# Hyperparameter tuning
LR_DEFAULT_CONFIG = {"learning_rate": LR_LEARNING_RATE, "batch_size": LR_BATCH_SIZE}
R_DEFAULT_CONFIG = {"learning_rate": R_LEARNING_RATE, "batch_size": R_BATCH_SIZE}
SEARCH_SPACE = {
    "learning_rate": tune.loguniform(1e-4, 1e-1),
    "batch_size": tune.choice([32, 64]),
}
TUNING_NUM_EPOCHS = 20
TUNING_NUM_SAMPLES = 10
