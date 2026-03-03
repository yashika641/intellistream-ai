import { useState } from "react";
import axios from "axios";

export default function Churn() {
  const [watchTime, setWatchTime] = useState("");
  const [subscriptionMonths, setSubscriptionMonths] = useState("");
  const [genreDiversity, setGenreDiversity] = useState("");
  const [watchlistItems, setWatchlistItems] = useState("");

  const [result, setResult] = useState<any>(null);
  const [loading, setLoading] = useState(false);

  const handlePredict = async () => {
    if (!watchTime || !subscriptionMonths) {
      return alert("Fill required fields");
    }

    setLoading(true);

    try {
      const response = await axios.post(
        "http://127.0.0.1:8000/churn_analytics/churn_analytics",
        [
          {
            watch_time: Number(watchTime),
            subscription_months: Number(subscriptionMonths),
            genre_diversity: Number(genreDiversity),
            watchlist_items: Number(watchlistItems),
          },
        ]
      );

      setResult(response.data);
    } catch (err: any) {
      alert(err.response?.data?.detail || "Something went wrong");
    }

    setLoading(false);
  };

  return (
    <div className="flex flex-col items-center justify-center min-h-screen bg-linear-to-br from-blue-50 to-blue-100 p-6">
      <h1 className="text-4xl font-bold mb-6 text-blue-800">
        🧠 Churn Analytics Dashboard
      </h1>

      <div className="bg-white p-6 rounded-2xl shadow-lg w-full max-w-lg space-y-3">
        <input
          type="number"
          placeholder="Watch Time (hrs/week)"
          value={watchTime}
          onChange={(e) => setWatchTime(e.target.value)}
          className="w-full border px-4 py-2 rounded-lg"
        />

        <input
          type="number"
          placeholder="Subscription Months"
          value={subscriptionMonths}
          onChange={(e) => setSubscriptionMonths(e.target.value)}
          className="w-full border px-4 py-2 rounded-lg"
        />

        <input
          type="number"
          placeholder="Genre Diversity"
          value={genreDiversity}
          onChange={(e) => setGenreDiversity(e.target.value)}
          className="w-full border px-4 py-2 rounded-lg"
        />

        <input
          type="number"
          placeholder="Watchlist Items"
          value={watchlistItems}
          onChange={(e) => setWatchlistItems(e.target.value)}
          className="w-full border px-4 py-2 rounded-lg"
        />

        <button
          onClick={handlePredict}
          disabled={loading}
          className="w-full bg-blue-600 text-white py-2 rounded-lg"
        >
          {loading ? "Analyzing..." : "Run Churn Analytics"}
        </button>
      </div>

      {result && (
        <div className="mt-8 w-full max-w-3xl bg-white shadow-md p-6 rounded-2xl">
          <h2 className="text-2xl font-bold text-blue-700 mb-4">
            📊 Churn Analysis Result
          </h2>

          <p><strong>Total Users:</strong> {result.total_users}</p>
          <p>
            <strong>Predicted Churn Rate:</strong>{" "}
            {result.predicted_churn_rate}%
          </p>
          <p>
            <strong>Retention Score:</strong>{" "}
            {result.retention_score}
          </p>
          <p>
            <strong>At Risk Users:</strong>{" "}
            {result.at_risk_users}
          </p>
          <p>
            <strong>Average Risk Probability:</strong>{" "}
            {result.average_risk_probability}%
          </p>
        </div>
      )}
    </div>
  );
}