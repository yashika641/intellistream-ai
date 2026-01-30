import { Navigation } from "../components/Navigation";
import { Card, CardContent, CardHeader, CardTitle } from "../components/ui/card";
import { motion } from "motion/react";
import { Upload, FileText, Sparkles, TrendingUp, BarChart3, Brain } from "lucide-react";
import { Page } from "../App";
import { Button } from "../components/ui/button";
import { useState } from "react";

interface ScriptPredictorPageProps {
  onNavigate: (page: Page) => void;
}

export function ScriptPredictorPage({ onNavigate }: ScriptPredictorPageProps) {
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [showResults, setShowResults] = useState(false);

  const handleAnalyze = () => {
    setIsAnalyzing(true);
    setTimeout(() => {
      setIsAnalyzing(false);
      setShowResults(true);
    }, 3000);
  };

  return (
    <div className="min-h-screen">
      <Navigation onNavigate={onNavigate} currentPage="Script Success Predictor" />
      
      <div className="max-w-7xl mx-auto px-6 py-12">
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6 }}
        >
          <div className="mb-12">
            <h1 className="mb-4 text-transparent bg-gradient-to-r from-blue-300 to-cyan-300 bg-clip-text">
              🎬 Script Success Predictor (NLP)
            </h1>
            <p className="text-slate-400 text-lg">
              Upload scripts or PDFs to get AI-based success probability & thematic breakdown
            </p>
          </div>

          {/* Upload Section */}
          <Card className="backdrop-blur-lg bg-slate-900/50 border border-blue-500/30 mb-8">
            <CardContent className="p-12">
              <div className="text-center">
                <div className="inline-flex items-center justify-center w-20 h-20 rounded-full bg-gradient-to-br from-blue-500/20 to-purple-500/20 border border-blue-500/30 mb-6">
                  <Upload className="size-10 text-blue-400" />
                </div>
                <h3 className="text-white mb-3">Upload Script File</h3>
                <p className="text-slate-400 mb-6">Supports PDF, DOCX, TXT formats</p>
                <div className="flex gap-3 justify-center">
                  <Button className="bg-gradient-to-r from-blue-600 to-purple-600 hover:from-blue-500 hover:to-purple-500">
                    <FileText className="size-4 mr-2" />
                    Choose File
                  </Button>
                  <Button 
                    variant="outline"
                    className="bg-slate-800/50 border-blue-500/30 text-blue-200 hover:bg-blue-500/10"
                    onClick={handleAnalyze}
                  >
                    <Sparkles className="size-4 mr-2" />
                    Analyze Sample Script
                  </Button>
                </div>
              </div>
            </CardContent>
          </Card>

          {isAnalyzing && (
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              className="mb-8"
            >
              <Card className="backdrop-blur-lg bg-slate-900/50 border border-purple-500/30">
                <CardContent className="p-8 text-center">
                  <div className="flex items-center justify-center gap-3 mb-4">
                    <motion.div
                      animate={{ rotate: 360 }}
                      transition={{ duration: 2, repeat: Infinity, ease: "linear" }}
                    >
                      <Brain className="size-8 text-purple-400" />
                    </motion.div>
                    <span className="text-white text-lg">Analyzing script with NLP models...</span>
                  </div>
                  <div className="max-w-md mx-auto">
                    <div className="h-2 bg-slate-800 rounded-full overflow-hidden">
                      <motion.div
                        className="h-full bg-gradient-to-r from-purple-500 to-blue-500"
                        initial={{ width: "0%" }}
                        animate={{ width: "100%" }}
                        transition={{ duration: 3 }}
                      />
                    </div>
                  </div>
                </CardContent>
              </Card>
            </motion.div>
          )}

          {showResults && (
            <motion.div
              initial={{ opacity: 0, y: 30 }}
              animate={{ opacity: 1, y: 0 }}
              className="space-y-6"
            >
              {/* Success Probability */}
              <Card className="backdrop-blur-lg bg-slate-900/50 border border-green-500/30 shadow-[0_0_30px_rgba(34,197,94,0.2)]">
                <CardHeader>
                  <CardTitle className="text-white flex items-center gap-2">
                    <TrendingUp className="size-5 text-green-400" />
                    Success Probability
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="text-center mb-6">
                    <div className="text-6xl mb-2">
                      <span className="text-transparent bg-gradient-to-r from-green-400 to-blue-400 bg-clip-text">
                        87.3%
                      </span>
                    </div>
                    <p className="text-slate-400">High probability of commercial success</p>
                  </div>
                  <div className="grid grid-cols-3 gap-4">
                    <div className="text-center p-4 rounded-lg bg-slate-800/50">
                      <p className="text-slate-400 mb-1">Box Office Potential</p>
                      <p className="text-green-400">$125M - $180M</p>
                    </div>
                    <div className="text-center p-4 rounded-lg bg-slate-800/50">
                      <p className="text-slate-400 mb-1">Audience Score</p>
                      <p className="text-blue-400">8.4 / 10</p>
                    </div>
                    <div className="text-center p-4 rounded-lg bg-slate-800/50">
                      <p className="text-slate-400 mb-1">Critical Rating</p>
                      <p className="text-purple-400">82%</p>
                    </div>
                  </div>
                </CardContent>
              </Card>

              {/* Thematic Analysis */}
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                <Card className="backdrop-blur-lg bg-slate-900/50 border border-purple-500/30">
                  <CardHeader>
                    <CardTitle className="text-white flex items-center gap-2">
                      <BarChart3 className="size-5 text-purple-400" />
                      Key Themes Detected
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="space-y-4">
                      {[
                        { theme: "Redemption & Personal Growth", strength: 92 },
                        { theme: "Family Dynamics", strength: 78 },
                        { theme: "Justice & Morality", strength: 71 },
                        { theme: "Sacrifice & Heroism", strength: 65 },
                      ].map((item, index) => (
                        <div key={index} className="space-y-2">
                          <div className="flex items-center justify-between text-sm">
                            <span className="text-slate-300">{item.theme}</span>
                            <span className="text-slate-400">{item.strength}%</span>
                          </div>
                          <div className="h-2 bg-slate-800 rounded-full overflow-hidden">
                            <div
                              className="h-full bg-gradient-to-r from-purple-500 to-blue-500"
                              style={{ width: `${item.strength}%` }}
                            />
                          </div>
                        </div>
                      ))}
                    </div>
                  </CardContent>
                </Card>

                <Card className="backdrop-blur-lg bg-slate-900/50 border border-blue-500/30">
                  <CardHeader>
                    <CardTitle className="text-white flex items-center gap-2">
                      <Brain className="size-5 text-blue-400" />
                      Genre Classification
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="space-y-4">
                      {[
                        { genre: "Drama", confidence: 89, color: "purple" },
                        { genre: "Thriller", confidence: 76, color: "blue" },
                        { genre: "Action", confidence: 62, color: "green" },
                        { genre: "Mystery", confidence: 54, color: "orange" },
                      ].map((item, index) => (
                        <div key={index} className="flex items-center justify-between p-4 rounded-lg bg-slate-800/50 border border-slate-700/50">
                          <div>
                            <p className="text-white mb-1">{item.genre}</p>
                            <div className="h-2 w-32 bg-slate-700 rounded-full overflow-hidden">
                              <div
                                className={`h-full bg-gradient-to-r from-${item.color}-500 to-${item.color}-600`}
                                style={{ width: `${item.confidence}%` }}
                              />
                            </div>
                          </div>
                          <span className="text-slate-400">{item.confidence}%</span>
                        </div>
                      ))}
                    </div>
                  </CardContent>
                </Card>
              </div>

              {/* AI Insights */}
              <Card className="backdrop-blur-lg bg-slate-900/50 border border-cyan-500/30">
                <CardHeader>
                  <CardTitle className="text-white flex items-center gap-2">
                    <Sparkles className="size-5 text-cyan-400" />
                    AI-Generated Insights
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="space-y-4">
                    <div className="p-4 rounded-lg bg-gradient-to-r from-cyan-500/10 to-blue-500/10 border border-cyan-500/30">
                      <p className="text-white mb-2">✨ Strong Character Development</p>
                      <p className="text-slate-400">The script features well-defined character arcs with clear motivations and emotional depth, particularly in the protagonist's journey.</p>
                    </div>
                    <div className="p-4 rounded-lg bg-gradient-to-r from-purple-500/10 to-pink-500/10 border border-purple-500/30">
                      <p className="text-white mb-2">🎯 Market Positioning</p>
                      <p className="text-slate-400">Best suited for theatrical release targeting 25-45 age demographic. Strong potential for streaming platform success.</p>
                    </div>
                    <div className="p-4 rounded-lg bg-gradient-to-r from-green-500/10 to-blue-500/10 border border-green-500/30">
                      <p className="text-white mb-2">📈 Commercial Viability</p>
                      <p className="text-slate-400">High commercial appeal with elements proven successful in recent box office hits. Recommended budget: $40M-$60M.</p>
                    </div>
                  </div>
                </CardContent>
              </Card>
            </motion.div>
          )}
        </motion.div>
      </div>
    </div>
  );
}
