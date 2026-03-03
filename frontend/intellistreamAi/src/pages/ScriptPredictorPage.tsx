/// <reference types="vite/client" />

import { Navigation } from "../components/Navigation";
import { Card, CardContent, CardHeader, CardTitle } from "../components/ui/card";
import { motion } from "motion/react";
import { Upload, FileText, Sparkles, TrendingUp, BarChart3, Brain } from "lucide-react";
import { Page } from "../App";
import { Button } from "../components/ui/button";
import { useState, useRef } from "react";
interface ScriptPredictorPageProps {
  onNavigate: (page: Page) => void;
}

export function ScriptPredictorPage({ onNavigate }: ScriptPredictorPageProps) {

  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [showResults, setShowResults] = useState(false);
  const [file, setFile] = useState<File | null>(null);
  const [result, setResult] = useState<any>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  // ------------------------------------------------
  // HANDLE FILE SELECT
  // ------------------------------------------------
  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      setFile(e.target.files[0]);
    }
  };

  // ------------------------------------------------
  // CALL BACKEND API
  // ------------------------------------------------
  const handleAnalyze = async () => {

    if (!file) {
      alert("Please upload a script file first");
      return;
    }

    setIsAnalyzing(true);
    setShowResults(false);

    try {
      const formData = new FormData();
      formData.append("file", file);
      const apiUrl = import.meta.env.VITE_API_URL || "http://127.0.0.1:8000";
      const response = await fetch(`${apiUrl}/script-success/analyze`, {
        method: "POST",
        body: formData
      });

      const data = await response.json();
      console.log("GENRE TYPE:", typeof data.genre_classification);
      console.log("IS ARRAY:", Array.isArray(data.genre_classification));
      console.log("GENRE VALUE:", data.genre_classification);
      setResult(data);
      setShowResults(true);

    } catch (error) {
      console.error("API Error:", error);
      alert("Failed to analyze script");
    }

    setIsAnalyzing(false);
  };

  return (
    <div className="min-h-screen">
      <Navigation onNavigate={onNavigate} currentPage="Script Success Predictor" />

      <div className="max-w-7xl mx-auto px-6 py-12">

        {/* Header */}
        <div className="mb-12">
          <h1 className="mb-4 text-transparent bg-gradient-to-r from-blue-300 to-cyan-300 bg-clip-text">
            🎬 Script Success Predictor (NLP)
          </h1>
          <p className="text-slate-400 text-lg">
            Upload scripts to get AI-based success probability & thematic breakdown
          </p>
        </div>

        {/* Upload Section */}
        <Card className="backdrop-blur-lg bg-slate-900/50 border border-blue-500/30 mb-8">
          <CardContent className="p-12 text-center">

            {/* Hidden File Input */}
            <input
              type="file"
              ref={fileInputRef}
              accept=".txt,.pdf,.docx"
              onChange={handleFileChange}
              className="hidden"
            />

            {/* Choose File Button */}
            <Button
              onClick={() => fileInputRef.current?.click()}
              className="bg-gradient-to-r from-blue-600 to-purple-600"
            >
              <Upload className="size-4 mr-2" />
              Choose File
            </Button>

            {file && (
              <p className="text-slate-400 mt-4">
                Selected: {file.name}
              </p>
            )}

            <div className="mt-6">
              <Button
                variant="outline"
                className="bg-slate-800/50 border-blue-500/30 text-blue-200"
                onClick={handleAnalyze}
              >
                <Sparkles className="size-4 mr-2" />
                Analyze Script
              </Button>
            </div>

          </CardContent>
        </Card>

        {/* Loading Animation */}
        {isAnalyzing && (
          <Card className="backdrop-blur-lg bg-slate-900/50 border border-purple-500/30 mb-8">
            <CardContent className="p-8 text-center">
              <motion.div
                animate={{ rotate: 360 }}
                transition={{ duration: 2, repeat: Infinity, ease: "linear" }}
              >
                <Brain className="size-8 text-purple-400 mx-auto mb-4" />
              </motion.div>
              <p className="text-white text-lg">
                Analyzing script with AI model...
              </p>
            </CardContent>
          </Card>
        )}

        {/* RESULTS */}
        {showResults && result && (

          <div className="space-y-6">

            {/* Success Probability */}
            <Card className="backdrop-blur-lg bg-slate-900/50 border border-green-500/30">
              <CardHeader>
                <CardTitle className="text-white flex items-center gap-2">
                  <TrendingUp className="size-5 text-green-400" />
                  Success Probability
                </CardTitle>
              </CardHeader>

              <CardContent>

                <div className="text-center mb-6">
                  <div className="text-6xl text-green-400">
                    {result.success_probability}%
                  </div>
                  <p className="text-slate-400">
                    {result.classification}
                  </p>
                </div>

                <div className="grid grid-cols-3 gap-4">

                  <div className="text-center p-4 rounded-lg bg-slate-800/50">
                    <p className="text-slate-400">Box Office Potential</p>
                    <p className="text-green-400">
                      {result.box_office_range}
                    </p>
                  </div>

                  <div className="text-center p-4 rounded-lg bg-slate-800/50">
                    <p className="text-slate-400">Audience Score</p>
                    <p className="text-blue-400">
                      {result.audience_score} / 10
                    </p>
                  </div>

                  <div className="text-center p-4 rounded-lg bg-slate-800/50">
                    <p className="text-slate-400">Critical Rating</p>
                    <p className="text-purple-400">
                      {result.critic_rating}%
                    </p>
                  </div>

                </div>
              </CardContent>
            </Card>

            {/* THEMES + GENRE SIDE BY SIDE */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">

              {/* THEMES */}
              <Card className="backdrop-blur-lg bg-slate-900/50 border border-purple-500/30">
                <CardHeader>
                  <CardTitle className="text-white">
                    Key Themes Detected
                  </CardTitle>
                </CardHeader>

                <CardContent>
                  {result.key_themes.map((theme: any, index: number) => (
                    <div key={index} className="mb-4">
                      <div className="flex justify-between text-sm text-slate-300">
                        <span>{theme.theme}</span>
                        <span>{theme.score}%</span>
                      </div>
                      <div className="h-2 bg-slate-800 rounded-full">
                        <div
                          className="h-full bg-gradient-to-r from-purple-500 to-blue-500"
                          style={{ width: `${theme.score}%` }}
                        />
                      </div>
                    </div>
                  ))}
                </CardContent>
              </Card>

              {/* GENRE CLASSIFICATION */}
              <Card className="backdrop-blur-lg bg-slate-900/50 border border-blue-500/30">
                <CardHeader>
                  <CardTitle className="text-white flex items-center gap-2">
                    <BarChart3 className="size-5 text-blue-400" />
                    Genre Classification
                  </CardTitle>
                </CardHeader>

                <CardContent>
                  {result.genre_classification &&
                    result.genre_classification.map((genre: any, index: number) => (
                      <div key={index} className="mb-4">
                        <div className="flex justify-between text-sm text-slate-300">
                          <span>{genre.genre}</span>
                          <span>{genre.score}%</span>
                        </div>
                        <div className="h-2 bg-slate-800 rounded-full">
                          <div
                            className="h-full bg-gradient-to-r from-blue-500 to-cyan-500"
                            style={{ width: `${genre.score}%` }}
                          />
                        </div>
                      </div>
                    ))}
                </CardContent>
              </Card>

            </div>

          </div>
        )}
      </div>
    </div>
  );
}