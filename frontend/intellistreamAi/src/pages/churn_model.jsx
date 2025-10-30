import { useState } from "react";
import axios from "axios";

export default function Churn() {
  const [customerId, setCustomerId] = useState("");
  const [result, setResult] = useState(null);
  const [topCustomers, setTopCustomers] = useState([]);
  const [loading, setLoading] = useState(false);

  const handlePredict = async () => {
    if (!customerId) return alert("Enter a Customer ID");
    setLoading(true);
    try {
      // Single customer prediction
      const res = await axios.get(`http://127.0.0.1:8000/api/predict_churn/${customerId}`);
      setResult(res.data);

      // Top 10 churners
      const topRes = await axios.get(`http://127.0.0.1:8000/api/top_churn_customers/?top_n=10`);
      setTopCustomers(topRes.data);

    } catch (err) {
      alert("Error: " + err.response?.data?.detail || "Something went wrong");
    }
    setLoading(false);
  };

  return (
    <div className="flex flex-col items-center justify-center min-h-screen bg-linear-to-br from-blue-50 to-blue-100 p-6">
      <h1 className="text-4xl font-bold mb-6 text-blue-800">🧠 Churn Prediction Dashboard</h1>

      <div className="bg-white p-6 rounded-2xl shadow-lg w-full max-w-lg">
        <input
          type="text"
          placeholder="Enter Customer ID"
          value={customerId}
          onChange={(e) => setCustomerId(e.target.value)}
          className="w-full border border-gray-300 rounded-lg px-4 py-2 mb-4 focus:ring-2 focus:ring-blue-500 outline-none"
        />
        <button
          onClick={handlePredict}
          disabled={loading}
          className="w-full bg-blue-600 text-white font-semibold py-2 rounded-lg hover:bg-blue-700 transition"
        >
          {loading ? "Predicting..." : "Predict Churn"}
        </button>
      </div>

      {result && (
        <div className="mt-8 w-full max-w-3xl bg-white shadow-md p-6 rounded-2xl">
          <h2 className="text-2xl font-bold text-blue-700 mb-4">
            Prediction for Customer ID {result.customer_id}
          </h2>
          <p className="text-lg mb-3">
            <strong>Churn Probability:</strong> {(result.churn_probability * 100).toFixed(2)}%
          </p>
          <p className="text-lg mb-6">
            <strong>Status:</strong>{" "}
            {result.churn_prediction ? (
              <span className="text-red-600 font-bold">Will Churn ❌</span>
            ) : (
              <span className="text-green-600 font-bold">Safe ✅</span>
            )}
          </p>

          <h3 className="text-xl font-semibold mb-2">🔥 Top 10 Customers Likely to Churn</h3>
          <table className="min-w-full border border-gray-200 rounded-lg text-sm">
            <thead className="bg-blue-100">
              <tr>
                <th className="px-4 py-2">Customer ID</th>
                <th className="px-4 py-2">Name</th>
                <th className="px-4 py-2">Churn Probability</th>
              </tr>
            </thead>
            <tbody>
              {topCustomers.map((cust, i) => (
                <tr key={i} className="text-center border-t">
                  <td className="px-4 py-2">{cust.customer_id}</td>
                  <td className="px-4 py-2">{cust.name}</td>
                  <td className="px-4 py-2 text-red-600 font-semibold">
                    {(cust.churn_probability * 100).toFixed(2)}%
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
