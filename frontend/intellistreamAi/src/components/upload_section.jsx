import { useState } from "react";

export default function UploadSection({ setPredictions }) {
  const [file, setFile] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleUpload = async () => {
    if (!file) return alert("Please upload a script file first!");
    setLoading(true);

    const formData = new FormData();
    formData.append("file", file);

    const res = await fetch("http://localhost:8000/predict", {
      method: "POST",
      body: formData,
    });

    const data = await res.json();
    setPredictions(data);
    setLoading(false);
  };

  return (
    <div className="flex flex-col items-center bg-gray-800 p-6 rounded-2xl shadow-lg w-full max-w-lg">
      <input
        type="file"
        accept=".txt"
        onChange={(e) => setFile(e.target.files[0])}
        className="mb-4 text-gray-200"
      />
      <button
        onClick={handleUpload}
        disabled={loading}
        className="bg-blue-600 hover:bg-blue-700 text-white px-6 py-2 rounded-xl shadow-md"
      >
        {loading ? "Predicting..." : "Upload & Predict"}
      </button>
    </div>
  );
}
