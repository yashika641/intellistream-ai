import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_recommenders as tfrs

# Load and preprocess data
def load_data():
    ratings = tfds.load("movielens/100k-ratings", split="train")
    movies = tfds.load("movielens/100k-movies", split="train")

    ratings = ratings.map(lambda x: {
        "user_id": x["user_id"],
        "movie_title": x["movie_title"]
    })

    movie_titles = movies.map(lambda x: x["movie_title"])

    unique_user_ids = set()
    for x in ratings.map(lambda x: x["user_id"]).batch(1000).unique().as_numpy_iterator():
        for user in x:
            unique_user_ids.add(user.decode("utf-8"))

    unique_movie_titles = set()
    for x in movie_titles.batch(1000).unique().as_numpy_iterator():
        for title in x:
            unique_movie_titles.add(title.decode("utf-8"))

    return ratings, movie_titles, sorted(unique_user_ids), sorted(unique_movie_titles)

# User and movie models
def build_user_model(user_ids):
    return tf.keras.Sequential([
        tf.keras.layers.StringLookup(vocabulary=user_ids, mask_token=None),
        tf.keras.layers.Embedding(input_dim=len(user_ids) + 1, output_dim=32)
    ])

def build_movie_model(movie_titles):
    return tf.keras.Sequential([
        tf.keras.layers.StringLookup(vocabulary=movie_titles, mask_token=None),
        tf.keras.layers.Embedding(input_dim=len(movie_titles) + 1, output_dim=32)
    ])

# Full retrieval model
class MovielensModel(tfrs.models.Model):
    def __init__(self, user_model, movie_model, task):
        super().__init__()
        self.user_model = user_model
        self.movie_model = movie_model
        self.task = task

    def compute_loss(self, features, training=False):
        user_embeddings = self.user_model(features["user_id"])
        movie_embeddings = self.movie_model(features["movie_title"])
        return self.task(user_embeddings, movie_embeddings)

# Main driver
def main():
    print("⏳ Loading data...")
    ratings, movie_titles_ds, unique_user_ids, unique_movie_titles = load_data()

    print("✅ Building models...")
    user_model = build_user_model(unique_user_ids)
    movie_model = build_movie_model(unique_movie_titles)

    retrieval_task = tfrs.tasks.Retrieval(
        metrics=tfrs.metrics.FactorizedTopK(
            candidates=movie_titles_ds.batch(128).map(lambda x: (x, movie_model(x)))
        )
    )

    model = MovielensModel(user_model, movie_model, retrieval_task)
    model.compile(optimizer=tf.keras.optimizers.Adagrad(0.5))

    train_data = ratings.shuffle(100_000).batch(4096).cache()

    print("🔧 Training model...")
    model.fit(train_data, epochs=3)

    print("🔍 Building retrieval index...")
    index = tfrs.layers.factorized_top_k.BruteForce(model.user_model)
    index.index_from_dataset(
        movie_titles_ds.map(lambda title: (title, movie_model(title))).batch(100)
    )

    user_id = "42"
    print(f"\n🎯 Top 5 movie recommendations for user {user_id}:")
    _, titles = index(tf.constant([user_id]))
    for i, title in enumerate(titles[0, :5].numpy()):
        print(f"{i + 1}. {title.decode('utf-8')}")

if __name__ == "__main__":
    main()
