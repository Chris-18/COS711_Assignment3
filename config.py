# Training Parameters
NUM_CLASSES = 1
LEARNING_RATE = 0.001
BATCH_SIZE = 64
NUM_EPOCHS = 50
NUM_WORKERS = 9

# Dataset Size Parameters
# NB! The train, validation and test sizes must add up to the input size exactly.
INPUT_SIZE = 26000
TRAIN_SIZE = 24000
VALIDATION_SIZE = 1000
TEST_SIZE = 1000

#Not deleting above - percentage instead
TRAIN_PERCENTAGE= 0.8
VALIDATION_PERCENTAGE= 0.1
TEST_PERCENTAGE = 0.1