import { ArrowLeft, Home, Menu } from "lucide-react";
import { Button } from "./ui/button";
import { Page } from "../App";
import { useState } from "react";

interface NavigationProps {
  onNavigate: (page: Page) => void;
  currentPage?: string;
}

export function Navigation({ onNavigate, currentPage }: NavigationProps) {
  const [isMenuOpen, setIsMenuOpen] = useState(false);

  return (
    <nav className="sticky top-0 z-50 backdrop-blur-lg bg-slate-900/80 border-b border-slate-800/50">
      <div className="max-w-7xl mx-auto px-6 py-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-4">
            <Button
              variant="ghost"
              size="sm"
              onClick={() => onNavigate("home")}
              className="text-purple-400 hover:text-purple-300 hover:bg-purple-500/10"
            >
              <ArrowLeft className="size-4 mr-2" />
              Back
            </Button>
            {currentPage && (
              <span className="text-slate-400">
                / <span className="text-white">{currentPage}</span>
              </span>
            )}
          </div>

          <div className="flex items-center gap-2">
            <Button
              variant="ghost"
              size="sm"
              onClick={() => onNavigate("home")}
              className="text-slate-400 hover:text-white hover:bg-slate-800"
            >
              <Home className="size-4 mr-2" />
              Home
            </Button>
            <Button
              variant="ghost"
              size="sm"
              onClick={() => setIsMenuOpen(!isMenuOpen)}
              className="text-slate-400 hover:text-white hover:bg-slate-800"
            >
              <Menu className="size-4" />
            </Button>
          </div>
        </div>

        {isMenuOpen && (
          <div className="mt-4 p-4 rounded-lg bg-slate-800/50 border border-slate-700">
            <div className="grid grid-cols-2 gap-2">
              <Button
                variant="ghost"
                onClick={() => { onNavigate("churn"); setIsMenuOpen(false); }}
                className="justify-start text-slate-300 hover:text-white hover:bg-purple-500/10"
              >
                👥 Churn Analytics
              </Button>
              <Button
                variant="ghost"
                onClick={() => { onNavigate("script"); setIsMenuOpen(false); }}
                className="justify-start text-slate-300 hover:text-white hover:bg-blue-500/10"
              >
                🎬 Script Predictor
              </Button>
              <Button
                variant="ghost"
                onClick={() => { onNavigate("recommendation"); setIsMenuOpen(false); }}
                className="justify-start text-slate-300 hover:text-white hover:bg-pink-500/10"
              >
                🎧 Recommendations
              </Button>
              <Button
                variant="ghost"
                onClick={() => { onNavigate("stock"); setIsMenuOpen(false); }}
                className="justify-start text-slate-300 hover:text-white hover:bg-green-500/10"
              >
                📈 Stock Forecast
              </Button>
              <Button
                variant="ghost"
                onClick={() => { onNavigate("sentiment"); setIsMenuOpen(false); }}
                className="justify-start text-slate-300 hover:text-white hover:bg-orange-500/10"
              >
                📰 Sentiment Analysis
              </Button>
              <Button
                variant="ghost"
                onClick={() => { onNavigate("explainable"); setIsMenuOpen(false); }}
                className="justify-start text-slate-300 hover:text-white hover:bg-cyan-500/10"
              >
                💡 Explainable AI
              </Button>
            </div>
          </div>
        )}
      </div>
    </nav>
  );
}
