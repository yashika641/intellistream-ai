import { Navigation } from "../components/Navigation";
import { Card, CardContent, CardHeader, CardTitle } from "../components/ui/card";
import { motion } from "motion/react";
import { Brain, Info, BarChart3, Activity, Target } from "lucide-react";
import { Page } from "../App";

interface ExplainableAIPageProps {
  onNavigate: (page: Page) => void;
}

const shapValues = [
  { feature: "Watch Time (hrs/week)", value: 0.42, impact: "positive" },
  { feature: "Content in Watchlist", value: 0.35, impact: "positive" },
  { feature: "Subscription Age", value: -0.28, impact: "negative" },
  { feature: "Genre Diversity Score", value: 0.22, impact: "positive" },
  { feature: "Average Session Length", value: 0.18, impact: "positive" },
  { feature: "Days Since Last Login", value: -0.31, impact: "negative" },
];

export function ExplainableAIPage({ onNavigate }: ExplainableAIPageProps) {
  return (
    <div className="min-h-screen">
      <Navigation onNavigate={onNavigate} currentPage="Explainable AI Insights" />
      
      <div className="max-w-7xl mx-auto px-6 py-12">
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6 }}
        >
          <div className="mb-12">
            <h1 className="mb-4 text-transparent bg-gradient-to-r from-cyan-300 to-purple-300 bg-clip-text">
              💡 Explainable AI Insights (SHAP/LIME)
            </h1>
            <p className="text-slate-400 text-lg">
              Understand why the model made a prediction with transparent AI explanations
            </p>
          </div>

          {/* Prediction Overview */}
          <Card className="backdrop-blur-lg bg-slate-900/50 border border-purple-500/30 mb-8 shadow-[0_0_30px_rgba(168,85,247,0.2)]">
            <CardHeader>
              <CardTitle className="text-white flex items-center gap-2">
                <Target className="size-5 text-purple-400" />
                Current Prediction Analysis
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-6">
                <div className="text-center p-6 rounded-lg bg-gradient-to-br from-red-500/10 to-orange-500/10 border border-red-500/30">
                  <p className="text-slate-400 mb-2">Churn Probability</p>
                  <p className="text-4xl mb-2">
                    <span className="text-transparent bg-gradient-to-r from-red-400 to-orange-400 bg-clip-text">
                      73.8%
                    </span>
                  </p>
                  <p className="text-red-400">High Risk</p>
                </div>
                <div className="text-center p-6 rounded-lg bg-gradient-to-br from-blue-500/10 to-cyan-500/10 border border-blue-500/30">
                  <p className="text-slate-400 mb-2">Model Confidence</p>
                  <p className="text-4xl mb-2">
                    <span className="text-transparent bg-gradient-to-r from-blue-400 to-cyan-400 bg-clip-text">
                      94.2%
                    </span>
                  </p>
                  <p className="text-blue-400">Very Confident</p>
                </div>
                <div className="text-center p-6 rounded-lg bg-gradient-to-br from-purple-500/10 to-pink-500/10 border border-purple-500/30">
                  <p className="text-slate-400 mb-2">User Segment</p>
                  <p className="text-4xl mb-2">
                    <span className="text-transparent bg-gradient-to-r from-purple-400 to-pink-400 bg-clip-text">
                      Casual
                    </span>
                  </p>
                  <p className="text-purple-400">Low Engagement</p>
                </div>
              </div>

              <div className="p-6 rounded-lg bg-gradient-to-r from-cyan-500/10 to-blue-500/10 border border-cyan-500/30">
                <div className="flex items-start gap-3">
                  <Info className="size-5 text-cyan-400 mt-1 flex-shrink-0" />
                  <div>
                    <p className="text-white mb-2">What does this mean?</p>
                    <p className="text-slate-400">
                      This user has a 73.8% probability of churning within the next 30 days. The model is highly confident in this prediction based on behavioral patterns. Key factors include low watch time and long subscription age.
                    </p>
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>

          {/* SHAP Values Explanation */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-8">
            <Card className="backdrop-blur-lg bg-slate-900/50 border border-blue-500/30">
              <CardHeader>
                <CardTitle className="text-white flex items-center gap-2">
                  <BarChart3 className="size-5 text-blue-400" />
                  SHAP Feature Importance
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  {shapValues.map((item, index) => (
                    <div key={index} className="space-y-2">
                      <div className="flex items-center justify-between text-sm">
                        <span className="text-slate-300">{item.feature}</span>
                        <span className={item.impact === 'positive' ? 'text-green-400' : 'text-red-400'}>
                          {item.value > 0 ? '+' : ''}{item.value.toFixed(2)}
                        </span>
                      </div>
                      <div className="relative h-3 bg-slate-800 rounded-full overflow-hidden">
                        <div className="absolute inset-0 flex items-center justify-center">
                          <div className="w-px h-full bg-slate-600" />
                        </div>
                        <div
                          className={`absolute h-full ${
                            item.impact === 'positive'
                              ? 'bg-gradient-to-r from-transparent to-green-500 left-1/2'
                              : 'bg-gradient-to-l from-transparent to-red-500 right-1/2'
                          }`}
                          style={{ 
                            width: `${Math.abs(item.value) * 100}%`,
                          }}
                        />
                      </div>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>

            <Card className="backdrop-blur-lg bg-slate-900/50 border border-purple-500/30">
              <CardHeader>
                <CardTitle className="text-white flex items-center gap-2">
                  <Brain className="size-5 text-purple-400" />
                  Feature Interactions
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="h-96 flex items-center justify-center border border-slate-800 rounded-lg bg-slate-950/50">
                  <div className="text-center">
                    <Activity className="size-12 mx-auto mb-3 text-purple-400" />
                    <p className="text-slate-500 mb-1">SHAP Dependence Plot</p>
                    <p className="text-slate-600 text-sm">Feature interaction visualization</p>
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>

          {/* Individual Prediction Explanation */}
          <Card className="backdrop-blur-lg bg-slate-900/50 border border-green-500/30">
            <CardHeader>
              <CardTitle className="text-white">LIME Local Explanation</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                <div className="p-4 rounded-lg bg-gradient-to-r from-red-500/10 to-orange-500/10 border border-red-500/30">
                  <div className="flex items-start gap-3 mb-3">
                    <div className="w-8 h-8 rounded-full bg-red-500/20 flex items-center justify-center flex-shrink-0">
                      <span className="text-red-400">−</span>
                    </div>
                    <div>
                      <p className="text-white mb-1">Low Watch Time (2.1 hrs/week)</p>
                      <p className="text-slate-400 text-sm">User watches significantly less than average (8.5 hrs/week). This is the strongest predictor of churn in this case.</p>
                    </div>
                  </div>
                </div>

                <div className="p-4 rounded-lg bg-gradient-to-r from-red-500/10 to-orange-500/10 border border-red-500/30">
                  <div className="flex items-start gap-3 mb-3">
                    <div className="w-8 h-8 rounded-full bg-red-500/20 flex items-center justify-center flex-shrink-0">
                      <span className="text-red-400">−</span>
                    </div>
                    <div>
                      <p className="text-white mb-1">Long Subscription Age (8 months)</p>
                      <p className="text-slate-400 text-sm">Users subscribed for 6+ months without increased engagement tend to churn more frequently.</p>
                    </div>
                  </div>
                </div>

                <div className="p-4 rounded-lg bg-gradient-to-r from-green-500/10 to-blue-500/10 border border-green-500/30">
                  <div className="flex items-start gap-3 mb-3">
                    <div className="w-8 h-8 rounded-full bg-green-500/20 flex items-center justify-center flex-shrink-0">
                      <span className="text-green-400">+</span>
                    </div>
                    <div>
                      <p className="text-white mb-1">Content in Watchlist (12 items)</p>
                      <p className="text-slate-400 text-sm">Having content saved shows intent to return, which slightly reduces churn probability.</p>
                    </div>
                  </div>
                </div>

                <div className="p-4 rounded-lg bg-gradient-to-r from-green-500/10 to-blue-500/10 border border-green-500/30">
                  <div className="flex items-start gap-3 mb-3">
                    <div className="w-8 h-8 rounded-full bg-green-500/20 flex items-center justify-center flex-shrink-0">
                      <span className="text-green-400">+</span>
                    </div>
                    <div>
                      <p className="text-white mb-1">Good Genre Diversity (Score: 7.2/10)</p>
                      <p className="text-slate-400 text-sm">User explores multiple genres, indicating broader platform engagement.</p>
                    </div>
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>
        </motion.div>
      </div>
    </div>
  );
}
