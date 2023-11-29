import nfp
import tensorflow as tf
from tensorflow.keras import layers


def build_model(preprocessor, atom_features, num_messages):
    """
    Function to build the FIA-GNN model with tensorflow.
    """
    # Define keras model
    atom = layers.Input(shape=[None], dtype=tf.int64, name="atom")
    atom_dist = layers.Input(shape=[None], dtype=tf.int64, name="dist_to_central_atom")
    bond = layers.Input(shape=[None], dtype=tf.int64, name="bond")
    connectivity = layers.Input(shape=[None, 2], dtype=tf.int64, name="connectivity")
    global_features = layers.Input(shape=[5], dtype=tf.float32, name="global_features")

    # Input tensors
    input_tensors = [atom, atom_dist, bond, connectivity, global_features]

    # Initialize the atom states
    atom_state_1 = layers.Embedding(
        preprocessor.atom_classes, atom_features, name="atom_embedding", mask_zero=True
    )(atom)

    # Initialize the bond states
    bond_state = layers.Embedding(
        preprocessor.bond_classes, atom_features, name="bond_embedding", mask_zero=True
    )(bond)

    # Initialize the atom_dist states
    atom_state_2 = layers.Embedding(
        int(preprocessor.max_atom_distance) + 1,
        atom_features,
        name="atom_dist_embedding",
        mask_zero=True,
    )(atom_dist)

    atom_state = layers.Concatenate(axis=-1)([atom_state_1, atom_state_2])
    atom_state = layers.Dense(512)(atom_state)
    atom_state = layers.Dense(256)(atom_state)
    atom_state = layers.Dense(128)(atom_state)

    global_features = layers.Dense(atom_features)(global_features)

    global_state = nfp.GlobalUpdate(units=atom_features, num_heads=1)(
        [atom_state, bond_state, connectivity]
    )

    # Do the message passing
    for _ in range(num_messages):
        new_bond_state = nfp.EdgeUpdate()(
            [atom_state, bond_state, connectivity, global_state]
        )
        bond_state = layers.Add()([bond_state, new_bond_state])

        new_atom_state = nfp.NodeUpdate()(
            [atom_state, bond_state, connectivity, global_state]
        )
        atom_state = layers.Add()([atom_state, new_atom_state])

        new_global_state = nfp.GlobalUpdate(units=atom_features, num_heads=1)(
            [atom_state, bond_state, connectivity, global_state]
        )
        global_state = layers.Add()([global_state, new_global_state])

    # Readout
    global_state = layers.Concatenate(axis=-1)([global_state, global_features])
    global_state = layers.Dense(128)(global_state)
    global_state = layers.Dense(64)(global_state)
    global_state = layers.Dense(32)(global_state)
    prop_pred = layers.Dense(1)(global_state)

    return tf.keras.Model(input_tensors, [prop_pred])
