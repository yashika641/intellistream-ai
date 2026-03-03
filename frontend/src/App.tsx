import { useState } from "react";
import { HomePage } from "./pages/HomePage";
import { ChurnAnalyticsPage } from "./pages/ChurnAnalyticsPage";
import { ScriptPredictorPage } from "./pages/ScriptPredictorPage";
import { RecommendationEnginePage } from "./pages/RecommendationEnginePage";
import { StockForecastPage } from "./pages/StockForecastPage";
import { SentimentAnalysisPage } from "./pages/SentimentAnalysisPage";
import { ExplainableAIPage } from "./pages/ExplainableAIPage";
import { UploadDataPage } from "./pages/UploadDataPage";

export type Page = 
  | "home" 
  | "churn" 
  | "script" 
  | "recommendation" 
  | "stock" 
  | "sentiment" 
  | "explainable"
  | "upload";

export default function App() {
  const [currentPage, setCurrentPage] = useState<Page>("home");

  const renderPage = () => {
    switch (currentPage) {
      case "home":
        return <HomePage onNavigate={setCurrentPage} />;
      case "churn":
        return <ChurnAnalyticsPage onNavigate={setCurrentPage} />;
      case "script":
        return <ScriptPredictorPage onNavigate={setCurrentPage} />;
      case "recommendation":
        return <RecommendationEnginePage onNavigate={setCurrentPage} />;
      case "stock":
        return <StockForecastPage onNavigate={setCurrentPage} />;
      case "sentiment":
        return <SentimentAnalysisPage onNavigate={setCurrentPage} />;
      case "explainable":
        return <ExplainableAIPage onNavigate={setCurrentPage} />;
      case "upload":
        return <UploadDataPage onNavigate={setCurrentPage} />;
      default:
        return <HomePage onNavigate={setCurrentPage} />;
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-950 via-purple-950 to-slate-900">
      {renderPage()}
    </div>
  );
}
