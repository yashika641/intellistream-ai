import { motion } from "motion/react";
import { Card, CardContent, CardHeader, CardTitle } from "./ui/card";
import { TrendingUp, TrendingDown, Users, Clock, BarChart3, PieChart, Activity } from "lucide-react";

const kpiData = [
  {
    title: "Churn Rate",
    value: "12.4%",
    change: "-2.1%",
    trend: "down",
    icon: TrendingDown,
    color: "green",
  },
  {
    title: "Avg Watch Time",
    value: "2.8h",
    change: "+0.5h",
    trend: "up",
    icon: Clock,
    color: "blue",
  },
  {
    title: "Active Users",
    value: "145K",
    change: "+12.3%",
    trend: "up",
    icon: Users,
    color: "purple",
  },
  {
    title: "Engagement Score",
    value: "87.5",
    change: "+5.2",
    trend: "up",
    icon: Activity,
    color: "pink",
  },
];

export function AnalyticsDashboard() {
  return (
    <section className="relative py-20 px-6">
      <div className="max-w-7xl mx-auto">
        <motion.div
          className="text-center mb-12"
          initial={{ opacity: 0, y: 30 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.6 }}
        >
          <h2 className="mb-4 text-transparent bg-gradient-to-r from-green-300 to-blue-300 bg-clip-text">
            Analytics Dashboard
          </h2>
          <p className="text-slate-400">Real-time insights and AI-powered predictions</p>
        </motion.div>

        {/* KPI Cards */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
          {kpiData.map((kpi, index) => {
            const Icon = kpi.icon;
            return (
              <motion.div
                key={index}
                initial={{ opacity: 0, y: 30 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ duration: 0.5, delay: index * 0.1 }}
              >
                <Card className="backdrop-blur-lg bg-slate-900/50 border border-slate-700/50 hover:border-purple-500/30 transition-all duration-300 group">
                  <CardContent className="p-6">
                    <div className="flex items-start justify-between mb-4">
                      <div className="flex-1">
                        <p className="text-slate-400 mb-2">{kpi.title}</p>
                        <p className="text-white">{kpi.value}</p>
                      </div>
                      <div
                        className={`p-3 rounded-lg bg-${kpi.color}-500/10 group-hover:scale-110 transition-transform duration-300`}
                      >
                        <Icon className="size-5 text-white" />
                      </div>
                    </div>
                    <div
                      className={`flex items-center gap-1 ${
                        kpi.trend === "up" ? "text-green-400" : "text-red-400"
                      }`}
                    >
                      {kpi.trend === "up" ? (
                        <TrendingUp className="size-4" />
                      ) : (
                        <TrendingDown className="size-4" />
                      )}
                      <span>{kpi.change} from last month</span>
                    </div>
                  </CardContent>
                </Card>
              </motion.div>
            );
          })}
        </div>

        {/* Charts Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Stock Chart */}
          <motion.div
            initial={{ opacity: 0, x: -30 }}
            whileInView={{ opacity: 1, x: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.6 }}
          >
            <Card className="backdrop-blur-lg bg-slate-900/50 border border-blue-500/30 shadow-[0_0_30px_rgba(59,130,246,0.1)]">
              <CardHeader>
                <CardTitle className="flex items-center gap-2 text-white">
                  <BarChart3 className="size-5 text-blue-400" />
                  Stock Price Forecast
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="h-64 flex items-center justify-center text-slate-500">
                  {/* Placeholder for chart - In production, use Recharts */}
                  <div className="text-center">
                    <TrendingUp className="size-12 mx-auto mb-2 text-blue-400" />
                    <p>Interactive stock chart with Prophet/LSTM predictions</p>
                  </div>
                </div>
              </CardContent>
            </Card>
          </motion.div>

          {/* Churn Analysis */}
          <motion.div
            initial={{ opacity: 0, x: 30 }}
            whileInView={{ opacity: 1, x: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.6 }}
          >
            <Card className="backdrop-blur-lg bg-slate-900/50 border border-purple-500/30 shadow-[0_0_30px_rgba(168,85,247,0.1)]">
              <CardHeader>
                <CardTitle className="flex items-center gap-2 text-white">
                  <PieChart className="size-5 text-purple-400" />
                  Churn Driver Analysis
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="h-64 flex items-center justify-center text-slate-500">
                  <div className="text-center">
                    <PieChart className="size-12 mx-auto mb-2 text-purple-400" />
                    <p>SHAP values & feature importance visualization</p>
                  </div>
                </div>
              </CardContent>
            </Card>
          </motion.div>

          {/* Recommendation Carousel */}
          <motion.div
            initial={{ opacity: 0, x: -30 }}
            whileInView={{ opacity: 1, x: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.6, delay: 0.2 }}
          >
            <Card className="backdrop-blur-lg bg-slate-900/50 border border-pink-500/30 shadow-[0_0_30px_rgba(236,72,153,0.1)]">
              <CardHeader>
                <CardTitle className="flex items-center gap-2 text-white">
                  <Activity className="size-5 text-pink-400" />
                  Top Recommendations
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-3">
                  {["Action Thriller - 94% Match", "Sci-Fi Drama - 89% Match", "Comedy Series - 85% Match"].map(
                    (item, index) => (
                      <div
                        key={index}
                        className="p-4 rounded-lg bg-slate-800/50 border border-slate-700/50 hover:border-pink-500/30 transition-all"
                      >
                        <div className="flex items-center justify-between">
                          <span className="text-slate-300">{item}</span>
                          <div className="h-2 w-20 bg-slate-700 rounded-full overflow-hidden">
                            <div
                              className="h-full bg-gradient-to-r from-pink-500 to-purple-500"
                              style={{ width: `${94 - index * 5}%` }}
                            />
                          </div>
                        </div>
                      </div>
                    )
                  )}
                </div>
              </CardContent>
            </Card>
          </motion.div>

          {/* NLP Topic Clusters */}
          <motion.div
            initial={{ opacity: 0, x: 30 }}
            whileInView={{ opacity: 1, x: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.6, delay: 0.2 }}
          >
            <Card className="backdrop-blur-lg bg-slate-900/50 border border-green-500/30 shadow-[0_0_30px_rgba(34,197,94,0.1)]">
              <CardHeader>
                <CardTitle className="flex items-center gap-2 text-white">
                  <BarChart3 className="size-5 text-green-400" />
                  Sentiment Analysis
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="h-64 flex items-center justify-center text-slate-500">
                  <div className="text-center">
                    <Activity className="size-12 mx-auto mb-2 text-green-400" />
                    <p>NLP topic clusters and sentiment distribution</p>
                  </div>
                </div>
              </CardContent>
            </Card>
          </motion.div>
        </div>
      </div>
    </section>
  );
}
