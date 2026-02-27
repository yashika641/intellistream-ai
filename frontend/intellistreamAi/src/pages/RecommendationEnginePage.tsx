import { useState } from "react";
import axios from "axios";
import { Navigation } from "../components/Navigation";
import { Card, CardContent, CardHeader, CardTitle } from "../components/ui/card";
import { motion } from "motion/react";
import { Star, Users, TrendingUp, Sparkles } from "lucide-react";
import { Page } from "../App";
import { Button } from "../components/ui/button";
import { Input } from "../components/ui/input";

interface RecommendationEnginePageProps {
  onNavigate: (page: Page) => void;
}

interface RecommendationItem {
  title: string;
  match_score: number;
  poster_url: string | null;
}

interface Metrics {
  watch_time_hours: number;
  top_genre: string;
  top_genre_percentage: number;
  similarity_score: number;
  model_accuracy: number;
}

export function RecommendationEnginePage({ onNavigate }: RecommendationEnginePageProps) {

  const [customerId, setCustomerId] = useState("");
  const [recommendations, setRecommendations] = useState<RecommendationItem[]>([]);
  const [metrics, setMetrics] = useState<Metrics | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const fetchRecommendations = async () => {
    if (!customerId) return alert("Enter Customer ID");

    try {
      setLoading(true);
      setError(null);

      const response = await axios.get(
        `http://localhost:8000/recommendations/${customerId}`
      );

      setRecommendations(response.data.recommendations);
      setMetrics(response.data.metrics);

    } catch (err: any) {
      setError(err.response?.data?.detail || "Failed to fetch recommendations");
      setRecommendations([]);
      setMetrics(null);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen">
      <Navigation onNavigate={onNavigate} currentPage="AI Recommendation Engine" />

      <div className="max-w-7xl mx-auto px-6 py-12">
        <motion.div initial={{ opacity: 0, y: 30 }} animate={{ opacity: 1, y: 0 }}>

          {/* Header */}
          <div className="mb-12">
            <h1 className="mb-4 text-transparent bg-gradient-to-r from-pink-300 to-purple-300 bg-clip-text">
              🎬 AI Recommendation Engine
            </h1>
            <p className="text-slate-400 text-lg">
              Hybrid AI Model: Neural Network + Collaborative Filtering
            </p>
          </div>

          {/* Customer Input MOVED ABOVE METRICS */}
          <Card className="mb-8 backdrop-blur-lg bg-slate-900/50 border border-purple-500/30">
            <CardContent className="p-6 flex flex-col md:flex-row gap-4 items-center">
              <Input
                placeholder="Enter Customer ID (e.g. U1)"
                value={customerId}
                onChange={(e) => setCustomerId(e.target.value)}
                className="bg-slate-800 border-slate-700 text-white"
              />
              <Button
                onClick={fetchRecommendations}
                className="bg-purple-600 hover:bg-purple-700 w-full md:w-auto"
              >
                {loading ? "Generating..." : "Generate Recommendations"}
              </Button>
            </CardContent>
          </Card>

          {error && <div className="text-red-400 mb-6">{error}</div>}

          {/* Metrics Loaded From API */}
          {metrics && (
            <div className="grid grid-cols-1 md:grid-cols-4 gap-6 mb-8">

              <Card className="backdrop-blur-lg bg-slate-900/50 border border-pink-500/30">
                <CardContent className="p-6">
                  <div className="flex justify-between mb-4">
                    <div>
                      <p className="text-slate-400 mb-2">Watch Time</p>
                      <p className="text-white">{metrics.watch_time_hours} hrs</p>
                    </div>
                    <TrendingUp className="size-5 text-pink-400" />
                  </div>
                  <div className="text-green-400">User engagement</div>
                </CardContent>
              </Card>

              <Card className="backdrop-blur-lg bg-slate-900/50 border border-purple-500/30">
                <CardContent className="p-6">
                  <div className="flex justify-between mb-4">
                    <div>
                      <p className="text-slate-400 mb-2">Top Genre</p>
                      <p className="text-white">{metrics.top_genre}</p>
                    </div>
                    <Star className="size-5 text-purple-400" />
                  </div>
                  <div className="text-slate-400">
                    {metrics.top_genre_percentage}% of content
                  </div>
                </CardContent>
              </Card>

              <Card className="backdrop-blur-lg bg-slate-900/50 border border-blue-500/30">
                <CardContent className="p-6">
                  <div className="flex justify-between mb-4">
                    <div>
                      <p className="text-slate-400 mb-2">Similarity Score</p>
                      <p className="text-white">{metrics.similarity_score}</p>
                    </div>
                    <Users className="size-5 text-blue-400" />
                  </div>
                  <div className="text-slate-400">User cluster confidence</div>
                </CardContent>
              </Card>

              <Card className="backdrop-blur-lg bg-slate-900/50 border border-green-500/30">
                <CardContent className="p-6">
                  <div className="flex justify-between mb-4">
                    <div>
                      <p className="text-slate-400 mb-2">Accuracy Rate</p>
                      <p className="text-white">{metrics.model_accuracy}%</p>
                    </div>
                    <Sparkles className="size-5 text-green-400" />
                  </div>
                  <div className="text-green-400">Hybrid model deployed</div>
                </CardContent>
              </Card>

            </div>
          )}

          {/* Recommendations */}
          {recommendations.length > 0 && (
            <Card className="backdrop-blur-lg bg-slate-900/50 border border-purple-500/30">
              <CardHeader>
                <CardTitle className="text-white">
                  Personalized For Customer: {customerId}
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  {recommendations.map((item, index) => (
                    <motion.div
                      key={index}
                      initial={{ opacity: 0, y: 20 }}
                      animate={{ opacity: 1, y: 0 }}
                      transition={{ delay: index * 0.1 }}
                      className="rounded-lg bg-slate-800/50 border border-slate-700/50 p-4"
                    >
                      <div className="flex gap-4">
                        <div className="w-32 h-48 bg-slate-700 rounded overflow-hidden">
                          {item.poster_url ? (
                            <img
                              src={item.poster_url}
                              alt={item.title}
                              className="w-full h-full object-cover"
                            />
                          ) : (
                            <div className="w-full h-full flex items-center justify-center text-slate-400">
                              No Image
                            </div>
                          )}
                        </div>

                        <div className="flex-1">
                          <h3 className="text-white mb-2">{item.title}</h3>
                          <div className="text-green-400 mb-2">
                            {item.match_score}% Match
                          </div>
                          <div className="text-slate-400 text-sm">
                            Hybrid AI prediction score
                          </div>
                        </div>
                      </div>
                    </motion.div>
                  ))}
                </div>
              </CardContent>
            </Card>
          )}

        </motion.div>
      </div>
    </div>
  );
}