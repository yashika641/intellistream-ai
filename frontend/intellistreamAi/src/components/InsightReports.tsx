import { motion } from "motion/react";
import { Card, CardContent } from "./ui/card";
import { Button } from "./ui/button";
import { FileText, Download, BarChart2, TrendingUp, Film, FileSpreadsheet } from "lucide-react";

const reports = [
  {
    icon: FileText,
    title: "Churn Insights Report",
    description: "Comprehensive analysis of churn drivers and retention strategies",
    format: "PDF",
    color: "purple",
  },
  {
    icon: TrendingUp,
    title: "Stock Forecast Report",
    description: "30-day predictions with confidence intervals and model metrics",
    format: "PDF",
    color: "blue",
  },
  {
    icon: Film,
    title: "Script Success Explanation",
    description: "AI-generated thematic breakdown and success probability analysis",
    format: "PDF",
    color: "pink",
  },
  {
    icon: FileSpreadsheet,
    title: "Full ML Pipeline Summary",
    description: "Complete model performance, features, and deployment details",
    format: "PDF",
    color: "green",
  },
];

export function InsightReports() {
  return (
    <section className="relative py-20 px-6">
      <div className="max-w-6xl mx-auto">
        <motion.div
          className="text-center mb-12"
          initial={{ opacity: 0, y: 30 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.6 }}
        >
          <h2 className="mb-4 text-transparent bg-gradient-to-r from-pink-300 to-purple-300 bg-clip-text">
            Download AI-Generated Reports
          </h2>
          <p className="text-slate-400 mb-2">Auto-generated. Enterprise-ready. Research-grade.</p>
        </motion.div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {reports.map((report, index) => {
            const Icon = report.icon;
            return (
              <motion.div
                key={index}
                initial={{ opacity: 0, y: 30 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ duration: 0.5, delay: index * 0.1 }}
              >
                <Card className="backdrop-blur-lg bg-slate-900/50 border border-slate-700/50 hover:border-purple-500/30 transition-all duration-300 group cursor-pointer">
                  <CardContent className="p-6">
                    <div className="flex items-start gap-4">
                      <div
                        className={`p-4 rounded-xl bg-${report.color}-500/10 border border-${report.color}-500/30 group-hover:scale-110 transition-transform duration-300`}
                      >
                        <Icon className="size-6 text-white" />
                      </div>
                      <div className="flex-1">
                        <div className="flex items-start justify-between mb-2">
                          <h3 className="text-white">{report.title}</h3>
                          <span className="px-2 py-1 rounded text-xs bg-slate-800 text-slate-400 border border-slate-700">
                            {report.format}
                          </span>
                        </div>
                        <p className="text-slate-400 mb-4">{report.description}</p>
                        <Button
                          size="sm"
                          className="bg-gradient-to-r from-purple-600 to-blue-600 hover:from-purple-500 hover:to-blue-500 text-white"
                        >
                          <Download className="size-4 mr-2" />
                          Download Report
                        </Button>
                      </div>
                    </div>
                  </CardContent>
                </Card>
              </motion.div>
            );
          })}
        </div>

        {/* Summary Stats */}
        <motion.div
          className="mt-12 grid grid-cols-1 md:grid-cols-3 gap-6"
          initial={{ opacity: 0, y: 30 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.6, delay: 0.4 }}
        >
          <div className="text-center p-6 rounded-xl bg-slate-900/50 border border-slate-700/50 backdrop-blur-lg">
            <BarChart2 className="size-8 mx-auto mb-3 text-purple-400" />
            <p className="text-slate-400 mb-1">Total Reports Generated</p>
            <p className="text-white">2,847</p>
          </div>
          <div className="text-center p-6 rounded-xl bg-slate-900/50 border border-slate-700/50 backdrop-blur-lg">
            <FileText className="size-8 mx-auto mb-3 text-blue-400" />
            <p className="text-slate-400 mb-1">Avg Processing Time</p>
            <p className="text-white">12.4 sec</p>
          </div>
          <div className="text-center p-6 rounded-xl bg-slate-900/50 border border-slate-700/50 backdrop-blur-lg">
            <TrendingUp className="size-8 mx-auto mb-3 text-green-400" />
            <p className="text-slate-400 mb-1">Model Accuracy</p>
            <p className="text-white">94.2%</p>
          </div>
        </motion.div>
      </div>
    </section>
  );
}
