import { Sparkles } from "lucide-react";
import { Page } from "../App";

interface FooterProps {
  onNavigate: (page: Page) => void;
}

export function Footer({ onNavigate }: FooterProps) {
  return (
    <footer className="relative py-12 px-6 border-t border-slate-800/50">
      <div className="max-w-7xl mx-auto">
        <div className="text-center mb-8">
          <div className="inline-flex items-center gap-2 mb-4 cursor-pointer" onClick={() => onNavigate("home")}>
            <Sparkles className="size-6 text-purple-400" />
            <span className="text-white">IntelliStreamAI</span>
          </div>
          <p className="text-slate-400 mb-6">
            Built for Next-Generation OTT + Financial Intelligence.
          </p>
        </div>

        <div className="flex flex-wrap justify-center gap-6 mb-8 text-slate-400">
          <a href="#" className="hover:text-purple-400 transition-colors" onClick={(e) => { e.preventDefault(); onNavigate("home"); }}>
            Home
          </a>
          <a href="#" className="hover:text-purple-400 transition-colors" onClick={(e) => { e.preventDefault(); const el = document.getElementById('modules-section'); el?.scrollIntoView({ behavior: 'smooth' }); }}>
            Modules
          </a>
          <a href="#" className="hover:text-purple-400 transition-colors">
            Docs
          </a>
          <a href="#" className="hover:text-purple-400 transition-colors">
            About
          </a>
          <a href="#" className="hover:text-purple-400 transition-colors">
            Contact
          </a>
        </div>

        <div className="text-center text-slate-500">
          <p className="mb-2">AI-powered by Python, TensorFlow, HuggingFace, Prophet, Streamlit, FastAPI</p>
          <p>© 2025 IntelliStreamAI. All rights reserved.</p>
        </div>
      </div>

      {/* Animated gradient line */}
      <div className="absolute bottom-0 left-0 right-0 h-px bg-gradient-to-r from-transparent via-purple-500 to-transparent opacity-50" />
    </footer>
  );
}