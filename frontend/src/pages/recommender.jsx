import { useState } from "react";
import axios from "axios";
import { gsap } from "gsap";

const TMDB_API_KEY = "eyJhbGciOiJIUzI1NiJ9.eyJhdWQiOiIyYzVjNDA0N2MyNmUwZmNlMzg5Y2E1ZmQ1YmUxMTA5MSIsIm5iZiI6MTc2MTAzNTY4Ni45NjEsInN1YiI6IjY4Zjc0NWE2M2I1ZDdhODk4MWY1YzZjYSIsInNjb3BlcyI6WyJhcGlfcmVhZCJdLCJ2ZXJzaW9uIjoxfQ.ts_TBS6JQRIyDVLUrZAsppVs0X8NpUo2jLRHczbgNp8";

export default function RecommenderUI() {
  const [customerId, setCustomerId] = useState("");
  const [recommendations, setRecommendations] = useState([]);
  const [movieTitlesBackend, setMovieTitlesBackend] = useState([]); // ✅ For debug
  const [loading, setLoading] = useState(false);

  const fetchRecommendations = async () => {
    if (!customerId) return alert("Please enter a customer ID");

    setLoading(true);
    try {
      // 1️⃣ Fetch recommended movie titles from your backend model
      const res = await axios.post("http://127.0.0.1:8000/api/recommend", { customer_id: customerId });
      const movieTitles = res.data.recommendations; // ["Inception", "Avatar", ...]
      setMovieTitlesBackend(movieTitles); // ✅ Show raw backend titles

      // 2️⃣ Fetch movie posters from TMDB API
      const movieData = await Promise.all(
        movieTitles.map(async (title) => {
          const tmdbRes = await axios.get(
            `https://api.themoviedb.org/3/search/movie?api_key=${TMDB_API_KEY}&query=${encodeURIComponent(title)}`
          );
          const movie = tmdbRes.data.results[0];
          return {
            title,
            poster: movie?.poster_path
              ? `https://image.tmdb.org/t/p/w500${movie.poster_path}`
              : "https://via.placeholder.com/500x750?text=No+Poster",
            overview: movie?.overview,
            rating: movie?.vote_average,
          };
        })
      );

      setRecommendations(movieData);
      setLoading(false);

      // 3️⃣ Animate the appearance
      gsap.fromTo(
        ".movie-card",
        { opacity: 0, y: 50 },
        { opacity: 1, y: 0, duration: 0.6, stagger: 0.1, ease: "power2.out" }
      );

    } catch (err) {
      console.error(err);
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-linear-to-b from-black to-gray-900 text-white flex flex-col items-center p-10">
      <h1 className="text-4xl md:text-6xl font-extrabold mb-10 text-transparent bg-clip-text bg-linear-to-r from-red-500 to-purple-500">
        🎬 Personalized Movie Recommender
      </h1>

      {/* Input Form */}
      <div className="flex gap-4 mb-10">
        <input
          type="text"
          placeholder="Enter Customer ID"
          value={customerId}
          onChange={(e) => setCustomerId(e.target.value)}
          className="px-4 py-2 rounded-lg text-black w-64"
        />
        <button
          onClick={fetchRecommendations}
          className="px-6 py-2 bg-red-600 hover:bg-red-700 rounded-lg transition-all"
        >
          Get Recommendations
        </button>
      </div>

      {/* Loading */}
      {loading && <p className="text-lg animate-pulse">Fetching your recommendations...</p>}

      {/* Debug: Show raw backend movie titles */}
      {movieTitlesBackend.length > 0 && (
        <div className="mb-4">
          <h3 className="font-semibold text-lg">Backend Recommendations:</h3>
          <ul className="list-disc list-inside text-gray-300">
            {movieTitlesBackend.map((title, i) => (
              <li key={i}>{title}</li>
            ))}
          </ul>
        </div>
      )}

      {/* Movie Results */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-6 mt-10">
        {recommendations.map((movie, index) => (
          <div
            key={index}
            className="movie-card bg-gray-800 rounded-xl overflow-hidden shadow-lg transform hover:scale-105 transition-all duration-300"
          >
            <img src={movie.poster} alt={movie.title} className="w-full h-80 object-cover" />
            <div className="p-3">
              <h3 className="text-lg font-semibold truncate">{movie.title}</h3>
              <p className="text-sm text-gray-400 line-clamp-2">{movie.overview}</p>
              <p className="mt-1 text-yellow-400">⭐ {movie.rating}</p>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
