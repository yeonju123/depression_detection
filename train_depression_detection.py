from transformers import TFAutoModel, AutoTokenizer
from datasets import load_dataset
from sklearn.model_selection import train_test_split
import tensorflow as tf
import argparse
import pandas as pd

# Load the depression dataset
def get_data(use_your_own_data=False, dataset_path=None):
    if use_your_own_data:
        # Load your own dataset with the assumption that your data is csv file with columns 'clean_text' and 'is_depression'
        dataset = pd.read_csv(dataset_path)
        pass
    else:
        # Load the default dataset
        dataset = load_dataset("ShreyaR/DepressionDetection")
    return dataset


def prepare_data(dataset):
    texts = dataset["clean_text"]
    labels = dataset["is_depression"]
    # Split the dataset into train and test
    train_texts, test_texts, train_labels, test_labels = train_test_split(
        texts, labels, test_size=0.2, random_state=42
    )
    # Split the train dataset into train and validation
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        train_texts, train_labels, test_size=0.25, random_state=42
    )
    return train_texts, val_texts, test_texts, train_labels, val_labels, test_labels

def main(args):
    # Load the pretrained model and tokenizer
    pretrained_model_name = "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)
    transformer_model = TFAutoModel.from_pretrained(pretrained_model_name)

    # Load the dataset
    dataset = get_data(args.use_your_own_data, args.dataset_path)
    train_texts, val_texts, test_texts, train_labels, val_labels, test_labels = prepare_data(dataset)
    # Tokenize the text
    train_encodings = tokenizer(train_texts, truncation=True, padding=True)
    val_encodings = tokenizer(val_texts, truncation=True, padding=True)
    test_encodings = tokenizer(test_texts, truncation=True, padding=True)

    # Convert to TensorFlow datasets
    train_dataset = tf.data.Dataset.from_tensor_slices(
        (dict(train_encodings), train_labels)
    )
    val_dataset = tf.data.Dataset.from_tensor_slices((dict(val_encodings), val_labels))
    test_dataset = tf.data.Dataset.from_tensor_slices((dict(test_encodings), test_labels))

    # Create the model
    input_layer = tf.keras.layers.Input(shape=(None,), dtype=tf.int32)
    # When you pass an input to a transformer model, 
    # it returns a tuple containing the last_hidden_state and pooler_output
    sequence_output = transformer_model(input_layer)[0]
    # We take the cls_token which is the first token of the output
    # cls_token has the aggregated representation of the whole sequence
    cls_token = sequence_output[:, 0, :]
    output_layer = tf.keras.layers.Dense(1, activation="sigmoid")(cls_token)
    model = tf.keras.Model(inputs=input_layer, outputs=output_layer)

    # Compile the model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=3e-5),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=["accuracy"],
    )

    # Train the model
    model.fit(train_dataset.batch(16), epochs=3, validation_data=val_dataset.batch(16))

    # Evaluate the model
    model.evaluate(test_dataset.batch(16))

    # Save the model
    model.save(args.output_model_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_your_own_data", action="store_true", help="Use your own data")
    parser.add_argument("--dataset_path", type=str, help="Path to your dataset")
    parser.add_argument("--output_model_path", type=str, help="Path to save the model")
    args = parser.parse_args()
    main(args)