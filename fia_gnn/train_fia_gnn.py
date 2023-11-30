import os
import time
import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa

from pathlib import Path
from fia_gnn_tf_model import build_model
from fia_gnn_preprocessing import (
    get_nfp_preprocessor,
    get_output_signature,
    get_padding_values,
)


@tf.function
def split_output(input_dict):
    """Formatting."""
    copied_dict = dict(input_dict)
    output = copied_dict.pop("output")
    return copied_dict, output


# For measuring the execution time.
start_time = time.time()

# Property to be learned
PROPERTY_NAME = "fia_gas-DSDBLYP"  # or alternatively "fia_solv-DSDBLYP"
MODEL_NAME = f"model_{PROPERTY_NAME}"

# Hyperparameter
LOSS = "mae"
EMBEDDING_VECTOR_LENGTH = 128
ROUNDS_OF_MESSAGE_PASSING = 6
BATCH_SIZE = 128
EPOCHS = 500
FIT_VERBOSITY = 1
CHECKPOINT_VERBOSITY = 0
SAVE_BEST_MODEL_ONLY = True
OPTIMIZER = tfa.optimizers.AdamW(
    learning_rate=tf.keras.optimizers.schedules.InverseTimeDecay(1e-3, 1, 1e-5),
    weight_decay=tf.keras.optimizers.schedules.InverseTimeDecay(1e-5, 1, 1e-5),
)

# Get preprocessor name
if PROPERTY_NAME == "fia_gas-DSDBLYP":
    PREPROCESSOR_NAME = "preprocessor_fia_gas"
if PROPERTY_NAME == "fia_solv-DSDBLYP":
    PREPROCESSOR_NAME = "preprocessor_fia_solv"

# Load the preprocessor
path_to_preprocessor = Path(
    os.getcwd(), f"preprocessed_data_{PROPERTY_NAME}/{PREPROCESSOR_NAME}.json"
)
preprocessor = get_nfp_preprocessor(path_to_preprocessor)

# Define output signatures and padding values
output_signature = get_output_signature(preprocessor)
output_signature['output'] = tf.TensorSpec(shape=(None,), dtype=tf.float32)
padding_values = get_padding_values(preprocessor)
padding_values['output'] = tf.constant(np.nan, dtype=tf.float32)

# Load the preprocessed data
data = pd.read_pickle(
    Path(
        f"preprocessed_data_{PROPERTY_NAME}/{PREPROCESSOR_NAME}_model_inputs.pkl",
    )
)

# Train data set
train = data[data.set_assignment == "train"]
train_dataset = (
    tf.data.Dataset.from_generator(
        lambda: iter(train.model_inputs),
        output_signature=output_signature,
    )
    .cache()
    .shuffle(buffer_size=len(train))
    .padded_batch(batch_size=BATCH_SIZE, padding_values=padding_values)
    .map(split_output)
    .prefetch(tf.data.experimental.AUTOTUNE)
)

# Validation data set
valid = data[data.set_assignment == "validate"]
valid_dataset = (
    tf.data.Dataset.from_generator(
        lambda: iter(valid.model_inputs),
        output_signature=output_signature,
    )
    .cache()
    .padded_batch(batch_size=BATCH_SIZE, padding_values=padding_values)
    .map(split_output)
    .prefetch(tf.data.experimental.AUTOTUNE)
)

# Build the model
model = build_model(
    preprocessor=preprocessor,
    atom_features=EMBEDDING_VECTOR_LENGTH,
    num_messages=ROUNDS_OF_MESSAGE_PASSING,
)

# Compile the model
model.compile(loss=LOSS, optimizer=OPTIMIZER)

# Outputs
if not os.path.exists(MODEL_NAME):
    os.makedirs(MODEL_NAME)

checkpoint = tf.keras.callbacks.ModelCheckpoint(
    MODEL_NAME, save_best_only=SAVE_BEST_MODEL_ONLY, verbose=CHECKPOINT_VERBOSITY
)
csv_logger = tf.keras.callbacks.CSVLogger(MODEL_NAME + "/log.csv")

print("####################")
print("# FIA-GNN Training #")
print("####################")
print()
print(f"Target property:                 {PROPERTY_NAME}")
print(f"Applied preprocessor:            {PREPROCESSOR_NAME} ({path_to_preprocessor})")
print()
print(f"Train data shape:                {train.shape}")
print(f"Validation data shape:           {valid.shape}")
print()
print(f"Size of embeddings:              {EMBEDDING_VECTOR_LENGTH}")
print(f"Rounds of message passing:       {ROUNDS_OF_MESSAGE_PASSING}")
print(f"Batch size for training:         {BATCH_SIZE}")
print(f"Number of epochs:                {EPOCHS}")
print(f"Loss for optimization:           {LOSS}")
print(f"Output directory:                {os.path.join(os.getcwd(), MODEL_NAME)}")
print(f"Only the best model is saved:    {SAVE_BEST_MODEL_ONLY}")
print(f"Name of the final best model:    {MODEL_NAME}")
print()

# Fit the model
print("Fitting the model now ...")
print(f"Start time: {time.asctime(time.localtime(time.time()))}")
print()

model.fit(
    train_dataset,
    validation_data=valid_dataset,
    epochs=EPOCHS,
    callbacks=[checkpoint, csv_logger],
    verbose=FIT_VERBOSITY,
)

print("Done.")
print()
print("Model training completed successfully.")
print(f"End time:      {time.asctime(time.localtime(time.time()))}")
print(f"Elapsed time:  {int(time.time() - start_time)} seconds")
print()
