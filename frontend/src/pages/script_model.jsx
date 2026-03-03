import React, { useState } from "react";
import axios from "axios";
import { motion } from "framer-motion";
import { FaUpload } from "react-icons/fa";
import { CircularProgressbar, buildStyles } from "react-circular-progressbar";
import "react-circular-progressbar/dist/styles.css";

export default function Script() {
  const [file, setFile] = useState(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);

  const handleFileChange = (e) => {
    setFile(e.target.files[0]);
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!file) return alert("Please upload a script (.txt) file first!");

    const formData = new FormData();
    formData.append("file", file);

    try {
      setLoading(true);
      setResult(null);

      const res = await axios.post("http://127.0.0.1:5000/predict/script", formData, {
        headers: { "Content-Type": "multipart/form-data" },
      });
      setResult(res.data);
    } catch (err) {
      console.error("Prediction failed:", err);
      alert("Something went wrong! Please check your backend connection.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen flex flex-col justify-center items-center bg-linear-to-br from-gray-900 via-black to-gray-800 text-white px-6 py-12">
      {/* Header */}
      <motion.div
        className="text-center mb-8"
        initial={{ opacity: 0, y: -30 }}
        animate={{ opacity: 1, y: 0 }}
      >
        <h1 className="text-5xl font-extrabold bg-linear-to-r from-blue-400 via-purple-500 to-pink-500 bg-clip-text text-transparent drop-shadow-lg">
          🎬 Script Success Predictor
        </h1>
        <p className="text-gray-400 mt-3 text-lg">Powered by BERT + Hybrid Deep Learning</p>
      </motion.div>

      {/* Upload Form */}
      <motion.form
        onSubmit={handleSubmit}
        className="bg-gray-800 rounded-2xl shadow-2xl border border-gray-700 p-10 w-full max-w-xl text-center"
        initial={{ opacity: 0, scale: 0.95 }}
        animate={{ opacity: 1, scale: 1 }}
      >
        <FaUpload className="text-6xl text-blue-400 mx-auto mb-4 animate-pulse" />
        <h2 className="text-2xl font-semibold mb-4">Upload your Movie Script (.txt)</h2>

        <input
          type="file"
          accept=".txt"
          onChange={handleFileChange}
          className="w-full mb-4 text-sm bg-gray-700 text-gray-300 border border-gray-600 rounded-lg cursor-pointer p-3"
        />

        <button
          type="submit"
          disabled={loading}
          className={`px-8 py-2 rounded-lg font-semibold transition-all duration-300 ${
            loading
              ? "bg-gray-600 cursor-not-allowed"
              : "bg-blue-600 hover:bg-blue-700 text-white"
          }`}
        >
          {loading ? "Analyzing..." : "Predict"}
        </button>
      </motion.form>

      {/* Loading Spinner */}
      {loading && (
        <div className="mt-10 flex flex-col items-center">
          <div className="animate-spin h-16 w-16 border-4 border-blue-500 border-t-transparent rounded-full"></div>
          <p className="mt-4 text-gray-300">Running AI models...</p>
        </div>
      )}

      {/* Results Section */}
      {result && (
        <motion.div
          className="bg-gray-900 mt-12 p-8 rounded-2xl border border-gray-700 shadow-xl w-full max-w-4xl"
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
        >
          <h2 className="text-3xl font-bold text-center text-green-400 mb-8">
            🧠 Prediction Results
          </h2>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
            {/* BERT DenseNN Model */}
            <div className="bg-gray-800 rounded-xl p-6 text-center border border-gray-700">
              <h3 className="text-xl font-semibold text-blue-400 mb-4">
                BERT + DenseNN Model
              </h3>
              <div className="flex justify-center mb-4">
                <div className="w-28 h-28">
                  <CircularProgressbar
                    value={result.bert_confidence}
                    text={`${result.bert_confidence}%`}
                    styles={buildStyles({
                      textColor: "#60a5fa",
                      pathColor: "#3b82f6",
                      trailColor: "#1f2937",
                    })}
                  />
                </div>
              </div>
              <p className="text-lg">
                🎯 Prediction:{" "}
                <span className="text-yellow-400 font-bold">{result.bert_pred}</span>
              </p>
            </div>

            {/* Hybrid Model */}
            <div className="bg-gray-800 rounded-xl p-6 text-center border border-gray-700">
              <h3 className="text-xl font-semibold text-purple-400 mb-4">Hybrid Model</h3>
              <div className="flex justify-center mb-4">
                <div className="w-28 h-28">
                  <CircularProgressbar
                    value={result.hybrid_confidence}
                    text={`${result.hybrid_confidence}%`}
                    styles={buildStyles({
                      textColor: "#c084fc",
                      pathColor: "#a855f7",
                      trailColor: "#1f2937",
                    })}
                  />
                </div>
              </div>
              <p className="text-lg">
                🎬 Prediction:{" "}
                <span className="text-yellow-400 font-bold">{result.hybrid_pred}</span>
              </p>
            </div>
          </div>

          {/* Additional Metadata */}
          <div className="mt-10 text-center text-gray-300 space-y-2">
            <h3 className="text-xl font-semibold text-gray-100 mb-2">
              🎥 Additional Insights
            </h3>
            <p>
              🧩 <span className="font-semibold text-white">Age Rating:</span>{" "}
              {result.age_rating}
            </p>
            <p>
              ⏱️ <span className="font-semibold text-white">Predicted Duration:</span>{" "}
              {result.predicted_duration} mins
            </p>
            <p>
              💬 <span className="font-semibold text-white">Sentiment:</span>{" "}
              {result.sentiment}
            </p>
          </div>
        </motion.div>
      )}
    </div>
  );
}
