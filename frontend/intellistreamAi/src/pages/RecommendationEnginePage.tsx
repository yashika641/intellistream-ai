import { Navigation } from "../components/Navigation";
import { Card, CardContent, CardHeader, CardTitle } from "../components/ui/card";
import { motion } from "motion/react";
import { Star, Users, TrendingUp, Sparkles, Play } from "lucide-react";
import { Page } from "../App";
import { Button } from "../components/ui/button";

interface RecommendationEnginePageProps {
  onNavigate: (page: Page) => void;
}

const recommendations = [
  {
    title: "Dark Horizon",
    genre: "Sci-Fi Thriller",
    match: 96,
    image: "https://images.unsplash.com/photo-1534447677768-be436bb09401?w=400",
    reasons: ["Similar viewing history", "Trending in your region", "High engagement rate"],
  },
  {
    title: "The Last Symphony",
    genre: "Drama",
    match: 94,
    image: "https://images.unsplash.com/photo-1478720568477-152d9b164e26?w=400",
    reasons: ["Matches your taste profile", "Award-winning content", "Popular among similar users"],
  },
  {
    title: "Code Red",
    genre: "Action",
    match: 91,
    image: "https://images.unsplash.com/photo-1485846234645-a62644f84728?w=400",
    reasons: ["Frequently binged", "New release", "High production value"],
  },
  {
    title: "Whispers in Time",
    genre: "Mystery",
    match: 89,
    image: "https://images.unsplash.com/photo-1518676590629-3dcbd9c5a5c9?w=400",
    reasons: ["Genre preference match", "Critical acclaim", "Recommended by AI"],
  },
];

export function RecommendationEnginePage({ onNavigate }: RecommendationEnginePageProps) {
  return (
    <div className="min-h-screen">
      <Navigation onNavigate={onNavigate} currentPage="AI Recommendation Engine" />
      
      <div className="max-w-7xl mx-auto px-6 py-12">
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6 }}
        >
          <div className="mb-12">
            <h1 className="mb-4 text-transparent bg-gradient-to-r from-pink-300 to-purple-300 bg-clip-text">
              🎧 AI Recommendation Engine
            </h1>
            <p className="text-slate-400 text-lg">
              Generate personalized content recommendations using behavior + content similarity
            </p>
          </div>

          {/* User Profile Overview */}
          <div className="grid grid-cols-1 md:grid-cols-4 gap-6 mb-8">
            <Card className="backdrop-blur-lg bg-slate-900/50 border border-pink-500/30">
              <CardContent className="p-6">
                <div className="flex items-start justify-between mb-4">
                  <div>
                    <p className="text-slate-400 mb-2">Watch Time</p>
                    <p className="text-white">127 hrs</p>
                  </div>
                  <div className="p-3 rounded-lg bg-pink-500/10">
                    <TrendingUp className="size-5 text-pink-400" />
                  </div>
                </div>
                <div className="text-green-400">This month</div>
              </CardContent>
            </Card>

            <Card className="backdrop-blur-lg bg-slate-900/50 border border-purple-500/30">
              <CardContent className="p-6">
                <div className="flex items-start justify-between mb-4">
                  <div>
                    <p className="text-slate-400 mb-2">Top Genre</p>
                    <p className="text-white">Sci-Fi</p>
                  </div>
                  <div className="p-3 rounded-lg bg-purple-500/10">
                    <Star className="size-5 text-purple-400" />
                  </div>
                </div>
                <div className="text-slate-400">42% of content</div>
              </CardContent>
            </Card>

            <Card className="backdrop-blur-lg bg-slate-900/50 border border-blue-500/30">
              <CardContent className="p-6">
                <div className="flex items-start justify-between mb-4">
                  <div>
                    <p className="text-slate-400 mb-2">Similarity Score</p>
                    <p className="text-white">0.89</p>
                  </div>
                  <div className="p-3 rounded-lg bg-blue-500/10">
                    <Users className="size-5 text-blue-400" />
                  </div>
                </div>
                <div className="text-slate-400">User cluster</div>
              </CardContent>
            </Card>

            <Card className="backdrop-blur-lg bg-slate-900/50 border border-green-500/30">
              <CardContent className="p-6">
                <div className="flex items-start justify-between mb-4">
                  <div>
                    <p className="text-slate-400 mb-2">Accuracy Rate</p>
                    <p className="text-white">92.4%</p>
                  </div>
                  <div className="p-3 rounded-lg bg-green-500/10">
                    <Sparkles className="size-5 text-green-400" />
                  </div>
                </div>
                <div className="text-green-400">Model performance</div>
              </CardContent>
            </Card>
          </div>

          {/* Personalized Recommendations */}
          <Card className="backdrop-blur-lg bg-slate-900/50 border border-purple-500/30 mb-8">
            <CardHeader>
              <CardTitle className="text-white flex items-center gap-2">
                <Star className="size-5 text-purple-400" />
                Personalized For You
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                {recommendations.map((item, index) => (
                  <motion.div
                    key={index}
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.5, delay: index * 0.1 }}
                    className="group relative overflow-hidden rounded-lg bg-slate-800/50 border border-slate-700/50 hover:border-purple-500/50 transition-all duration-300 cursor-pointer"
                  >
                    <div className="flex gap-4 p-4">
                      <div className="relative w-32 h-48 flex-shrink-0 rounded-lg overflow-hidden">
                        <img 
                          src={item.image} 
                          alt={item.title}
                          className="w-full h-full object-cover group-hover:scale-110 transition-transform duration-300"
                        />
                        <div className="absolute inset-0 bg-gradient-to-t from-black/80 to-transparent flex items-center justify-center opacity-0 group-hover:opacity-100 transition-opacity">
                          <Play className="size-12 text-white" />
                        </div>
                        <div className="absolute top-2 right-2 px-2 py-1 rounded bg-green-500 text-white text-xs">
                          {item.match}% Match
                        </div>
                      </div>
                      
                      <div className="flex-1">
                        <h3 className="text-white mb-2">{item.title}</h3>
                        <p className="text-slate-400 mb-4">{item.genre}</p>
                        
                        <div className="space-y-2">
                          <p className="text-slate-500 text-sm">Why we recommend:</p>
                          {item.reasons.map((reason, i) => (
                            <div key={i} className="flex items-center gap-2 text-sm">
                              <div className="w-1.5 h-1.5 rounded-full bg-purple-400" />
                              <span className="text-slate-400">{reason}</span>
                            </div>
                          ))}
                        </div>
                      </div>
                    </div>
                  </motion.div>
                ))}
              </div>
            </CardContent>
          </Card>

          {/* Model Performance */}
          <Card className="backdrop-blur-lg bg-slate-900/50 border border-blue-500/30">
            <CardHeader>
              <CardTitle className="text-white flex items-center gap-2">
                <Sparkles className="size-5 text-blue-400" />
                Recommendation Algorithm Performance
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                <div className="p-6 rounded-lg bg-gradient-to-br from-purple-500/10 to-blue-500/10 border border-purple-500/30">
                  <h3 className="text-white mb-2">Collaborative Filtering</h3>
                  <div className="text-3xl mb-2">
                    <span className="text-transparent bg-gradient-to-r from-purple-400 to-blue-400 bg-clip-text">
                      94.2%
                    </span>
                  </div>
                  <p className="text-slate-400">User-based similarity accuracy</p>
                </div>
                
                <div className="p-6 rounded-lg bg-gradient-to-br from-blue-500/10 to-cyan-500/10 border border-blue-500/30">
                  <h3 className="text-white mb-2">Content-Based</h3>
                  <div className="text-3xl mb-2">
                    <span className="text-transparent bg-gradient-to-r from-blue-400 to-cyan-400 bg-clip-text">
                      89.7%
                    </span>
                  </div>
                  <p className="text-slate-400">Genre & metadata matching</p>
                </div>
                
                <div className="p-6 rounded-lg bg-gradient-to-br from-pink-500/10 to-purple-500/10 border border-pink-500/30">
                  <h3 className="text-white mb-2">Hybrid Model</h3>
                  <div className="text-3xl mb-2">
                    <span className="text-transparent bg-gradient-to-r from-pink-400 to-purple-400 bg-clip-text">
                      96.8%
                    </span>
                  </div>
                  <p className="text-slate-400">Combined approach (deployed)</p>
                </div>
              </div>
            </CardContent>
          </Card>
        </motion.div>
      </div>
    </div>
  );
}
