import { motion } from "motion/react";
import { ModuleCard } from "./ModuleCard";
import { Users, Sparkles, Star, TrendingUp, Newspaper, Brain } from "lucide-react";
import { Page } from "../App";

const modules = [
  {
    emoji: "👥",
    title: "OTT Churn & Behavior Analytics",
    description: "Analyze churn drivers, content engagement, and retention signals.",
    icon: Users,
    color: "purple",
    page: "churn" as Page,
  },
  {
    emoji: "🎬",
    title: "Script Success Predictor (NLP)",
    description: "Upload scripts or PDFs to get AI-based success probability & thematic breakdown.",
    icon: Sparkles,
    color: "blue",
    page: "script" as Page,
  },
  {
    emoji: "🎧",
    title: "AI Recommendation Engine",
    description: "Generate personalized content recommendations using behavior + content similarity.",
    icon: Star,
    color: "pink",
    page: "recommendation" as Page,
  },
  {
    emoji: "📈",
    title: "Stock Market Trend Forecaster",
    description: "Forecast media stock prices using Prophet, ARIMA & LSTM.",
    icon: TrendingUp,
    color: "green",
    page: "stock" as Page,
  },
  {
    emoji: "📰",
    title: "Sentiment & Social Pulse (NLP + News)",
    description: "Analyze viewer sentiment and news headlines to detect market-moving buzz.",
    icon: Newspaper,
    color: "orange",
    page: "sentiment" as Page,
  },
  {
    emoji: "💡",
    title: "Explainable AI Insights (SHAP/LIME)",
    description: "Understand why the model made a prediction.",
    icon: Brain,
    color: "cyan",
    page: "explainable" as Page,
  },
];

interface ModulesSectionProps {
  onNavigate: (page: Page) => void;
}

export function ModulesSection({ onNavigate }: ModulesSectionProps) {
  return (
    <section id="modules-section" className="relative py-20 px-6">
      <div className="max-w-7xl mx-auto">
        <motion.div
          className="text-center mb-16"
          initial={{ opacity: 0, y: 30 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.6 }}
        >
          <h2 className="mb-4 text-transparent bg-gradient-to-r from-purple-300 to-blue-300 bg-clip-text">
            What would you like to analyze today?
          </h2>
        </motion.div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {modules.map((module, index) => (
            <ModuleCard key={index} module={module} index={index} onNavigate={onNavigate} />
          ))}
        </div>
      </div>
    </section>
  );
}