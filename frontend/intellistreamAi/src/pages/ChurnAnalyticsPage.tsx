import { Navigation } from "../components/Navigation";
import { Card, CardContent, CardHeader, CardTitle } from "../components/ui/card";
import { motion } from "motion/react";
import { TrendingDown, Users, Clock, AlertCircle, Target, BarChart3 } from "lucide-react";
import { Page } from "../App";
import { Button } from "../components/ui/button";

interface ChurnAnalyticsPageProps {
  onNavigate: (page: Page) => void;
}

export function ChurnAnalyticsPage({ onNavigate }: ChurnAnalyticsPageProps) {
  return (
    <div className="min-h-screen">
      <Navigation onNavigate={onNavigate} currentPage="Churn & Behavior Analytics" />
      
      <div className="max-w-7xl mx-auto px-6 py-12">
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6 }}
        >
          <div className="mb-12">
            <h1 className="mb-4 text-transparent bg-gradient-to-r from-purple-300 to-pink-300 bg-clip-text">
              👥 OTT Churn & Behavior Analytics
            </h1>
            <p className="text-slate-400 text-lg">
              Analyze churn drivers, content engagement, and retention signals with advanced ML models
            </p>
          </div>

          {/* KPI Grid */}
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
            <Card className="backdrop-blur-lg bg-slate-900/50 border border-red-500/30">
              <CardContent className="p-6">
                <div className="flex items-start justify-between mb-4">
                  <div>
                    <p className="text-slate-400 mb-2">Churn Rate</p>
                    <p className="text-white">12.4%</p>
                  </div>
                  <div className="p-3 rounded-lg bg-red-500/10">
                    <TrendingDown className="size-5 text-red-400" />
                  </div>
                </div>
                <div className="text-green-400">↓ 2.1% vs last month</div>
              </CardContent>
            </Card>

            <Card className="backdrop-blur-lg bg-slate-900/50 border border-blue-500/30">
              <CardContent className="p-6">
                <div className="flex items-start justify-between mb-4">
                  <div>
                    <p className="text-slate-400 mb-2">At-Risk Users</p>
                    <p className="text-white">1,847</p>
                  </div>
                  <div className="p-3 rounded-lg bg-blue-500/10">
                    <AlertCircle className="size-5 text-blue-400" />
                  </div>
                </div>
                <div className="text-red-400">↑ 5.3% vs last month</div>
              </CardContent>
            </Card>

            <Card className="backdrop-blur-lg bg-slate-900/50 border border-purple-500/30">
              <CardContent className="p-6">
                <div className="flex items-start justify-between mb-4">
                  <div>
                    <p className="text-slate-400 mb-2">Avg Engagement</p>
                    <p className="text-white">87.5%</p>
                  </div>
                  <div className="p-3 rounded-lg bg-purple-500/10">
                    <Target className="size-5 text-purple-400" />
                  </div>
                </div>
                <div className="text-green-400">↑ 3.2% vs last month</div>
              </CardContent>
            </Card>

            <Card className="backdrop-blur-lg bg-slate-900/50 border border-green-500/30">
              <CardContent className="p-6">
                <div className="flex items-start justify-between mb-4">
                  <div>
                    <p className="text-slate-400 mb-2">Retention Score</p>
                    <p className="text-white">92.1</p>
                  </div>
                  <div className="p-3 rounded-lg bg-green-500/10">
                    <Users className="size-5 text-green-400" />
                  </div>
                </div>
                <div className="text-green-400">↑ 1.8 vs last month</div>
              </CardContent>
            </Card>
          </div>

          {/* Main Analysis Cards */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-8">
            <Card className="backdrop-blur-lg bg-slate-900/50 border border-purple-500/30">
              <CardHeader>
                <CardTitle className="text-white flex items-center gap-2">
                  <BarChart3 className="size-5 text-purple-400" />
                  Top Churn Drivers
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  {[
                    { driver: "Low watch time (< 5hrs/week)", impact: 85, color: "red" },
                    { driver: "No content in watchlist", impact: 72, color: "orange" },
                    { driver: "Subscription > 6 months", impact: 68, color: "yellow" },
                    { driver: "Minimal genre diversity", impact: 54, color: "blue" },
                  ].map((item, index) => (
                    <div key={index} className="space-y-2">
                      <div className="flex items-center justify-between text-sm">
                        <span className="text-slate-300">{item.driver}</span>
                        <span className="text-slate-400">{item.impact}% impact</span>
                      </div>
                      <div className="h-2 bg-slate-800 rounded-full overflow-hidden">
                        <div
                          className={`h-full bg-gradient-to-r from-${item.color}-500 to-${item.color}-600`}
                          style={{ width: `${item.impact}%` }}
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
                  <Clock className="size-5 text-blue-400" />
                  User Behavior Segments
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  {[
                    { segment: "Power Users", users: "34.2k", churn: "3.1%", color: "green" },
                    { segment: "Regular Viewers", users: "87.5k", churn: "8.4%", color: "blue" },
                    { segment: "Casual Users", users: "45.8k", churn: "18.7%", color: "orange" },
                    { segment: "Inactive", users: "12.3k", churn: "42.3%", color: "red" },
                  ].map((item, index) => (
                    <div key={index} className="flex items-center justify-between p-4 rounded-lg bg-slate-800/50 border border-slate-700/50">
                      <div>
                        <p className="text-white mb-1">{item.segment}</p>
                        <p className="text-slate-400">{item.users} users</p>
                      </div>
                      <div className={`text-${item.color}-400`}>
                        {item.churn} churn
                      </div>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          </div>

          {/* Recommendations */}
          <Card className="backdrop-blur-lg bg-slate-900/50 border border-green-500/30">
            <CardHeader>
              <CardTitle className="text-white">💡 AI-Powered Recommendations</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                <div className="p-6 rounded-lg bg-gradient-to-br from-purple-500/10 to-blue-500/10 border border-purple-500/30">
                  <h3 className="text-white mb-2">Personalization Campaign</h3>
                  <p className="text-slate-400 mb-4">Target 1,847 at-risk users with personalized content recommendations</p>
                  <Button size="sm" className="bg-purple-600 hover:bg-purple-500">Launch Campaign</Button>
                </div>
                <div className="p-6 rounded-lg bg-gradient-to-br from-green-500/10 to-blue-500/10 border border-green-500/30">
                  <h3 className="text-white mb-2">Engagement Boost</h3>
                  <p className="text-slate-400 mb-4">Send push notifications for new releases in favorite genres</p>
                  <Button size="sm" className="bg-green-600 hover:bg-green-500">Activate</Button>
                </div>
                <div className="p-6 rounded-lg bg-gradient-to-br from-orange-500/10 to-red-500/10 border border-orange-500/30">
                  <h3 className="text-white mb-2">Win-Back Strategy</h3>
                  <p className="text-slate-400 mb-4">Offer limited-time discount to inactive segment</p>
                  <Button size="sm" className="bg-orange-600 hover:bg-orange-500">Configure</Button>
                </div>
              </div>
            </CardContent>
          </Card>
        </motion.div>
      </div>
    </div>
  );
}
