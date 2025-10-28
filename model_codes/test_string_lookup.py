import tensorflow as tf
import tensorflow_recommenders as tfrs
import tensorflow_datasets as tfds

# Load a small Movielens dataset
ratings = tfds.load("movielens/100k-ratings", split="train")
ratings = ratings.map(lambda x: {
    "user_id": x["user_id"],
    "movie_title": x["movie_title"]
})

# Get unique values
movie_titles = ratings.map(lambda x: x["movie_title"])
user_ids = ratings.map(lambda x: x["user_id"])

unique_movie_titles = movie_titles.batch(1000).unique()
unique_user_ids = user_ids.batch(1000).unique()

movie_titles_list = [x.decode("utf-8") for x in next(iter(unique_movie_titles.as_numpy_iterator()))]
user_ids_list = [x.decode("utf-8") for x in next(iter(unique_user_ids.as_numpy_iterator()))]

# Define models
embedding_dim = 32

user_model = tf.keras.Sequential([
    tf.keras.layers.StringLookup(vocabulary=user_ids_list, mask_token=None),
    tf.keras.layers.Embedding(input_dim=len(user_ids_list) + 1, output_dim=embedding_dim)
])

movie_model = tf.keras.Sequential([
    tf.keras.layers.StringLookup(vocabulary=movie_titles_list, mask_token=None),
    tf.keras.layers.Embedding(input_dim=len(movie_titles_list) + 1, output_dim=embedding_dim)
])

# FactorizedTopK with proper (id, embedding) structure
candidate_dataset = tf.data.Dataset.from_tensor_slices(movie_titles_list).map(
    lambda title: (title, movie_model(tf.convert_to_tensor([title]))[0])
).batch(128)

retrieval_task = tfrs.tasks.Retrieval(
    metrics=tfrs.metrics.FactorizedTopK(candidates=candidate_dataset)
)

# Build TFRS Model
class MovielensModel(tfrs.Model):
    def __init__(self):
        super().__init__()
        self.user_model = user_model
        self.movie_model = movie_model
        self.task = retrieval_task

    def compute_loss(self, features, training=False):
        user_embeddings = self.user_model(features["user_id"])
        movie_embeddings = self.movie_model(features["movie_title"])
        return self.task(user_embeddings, movie_embeddings)

# Prepare dataset
cached = ratings.shuffle(100_000).batch(256).cache()

# Train the model
model = MovielensModel()
model.compile(optimizer=tf.keras.optimizers.Adagrad(0.5))
model.fit(cached, epochs=3)
