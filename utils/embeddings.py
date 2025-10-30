import tensorflow as tf

def get_bert_embedding(text, tokenizer, model):
    """
    Returns the mean-pooled BERT embedding for input text.
    """
    inputs = tokenizer(
        text,
        return_tensors='tf',
        truncation=True,
        padding=True,
        max_length=512
    )
    outputs = model(**inputs)
    return tf.reduce_mean(outputs.last_hidden_state, axis=1).numpy()
