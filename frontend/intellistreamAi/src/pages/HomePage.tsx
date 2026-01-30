import { HeroSection } from "../components/HeroSection";
import { ModulesSection } from "../components/ModulesSection";
import { UploadPanel } from "../components/UploadPanel";
import { AIAssistant } from "../components/AIAssistant";
import { AnalyticsDashboard } from "../components/AnalyticsDashboard";
import { InsightReports } from "../components/InsightReports";
import { Footer } from "../components/Footer";
import { Page } from "../App";

interface HomePageProps {
  onNavigate: (page: Page) => void;
}

export function HomePage({ onNavigate }: HomePageProps) {
  return (
    <>
      <HeroSection onNavigate={onNavigate} />
      <ModulesSection onNavigate={onNavigate} />
      <UploadPanel onNavigate={onNavigate} />
      <AIAssistant />
      <AnalyticsDashboard />
      <InsightReports />
      <Footer onNavigate={onNavigate} />
    </>
  );
}
