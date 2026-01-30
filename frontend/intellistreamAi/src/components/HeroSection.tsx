import { motion } from "motion/react";
import { Button } from "./ui/button";
import { Sparkles, Upload, Layers } from "lucide-react";
import { Page } from "../App";
import { useState } from "react";

interface HeroSectionProps {
  onNavigate: (page: Page) => void;
}

export function HeroSection({ onNavigate }: HeroSectionProps) {
  const [isAuthenticated, setIsAuthenticated] = useState(false);
  const [showAuthModal, setShowAuthModal] = useState<"signin" | "signup" | null>(null);

  return (
    <section className="relative min-h-screen flex items-center justify-center overflow-hidden px-6 py-20">
      {/* Animated Background Elements */}
      <div className="absolute inset-0 overflow-hidden">
        <motion.div
          className="absolute top-20 left-10 w-72 h-72 bg-purple-500/20 rounded-full blur-3xl"
          animate={{
            scale: [1, 1.2, 1],
            opacity: [0.3, 0.5, 0.3],
          }}
          transition={{
            duration: 8,
            repeat: Infinity,
            ease: "easeInOut",
          }}
        />
        <motion.div
          className="absolute bottom-20 right-10 w-96 h-96 bg-blue-500/20 rounded-full blur-3xl"
          animate={{
            scale: [1, 1.3, 1],
            opacity: [0.3, 0.5, 0.3],
          }}
          transition={{
            duration: 10,
            repeat: Infinity,
            ease: "easeInOut",
          }}
        />

        {/* Grid Pattern */}
        <div className="absolute inset-0 bg-[linear-gradient(to_right,#4f4f4f12_1px,transparent_1px),linear-gradient(to_bottom,#4f4f4f12_1px,transparent_1px)] bg-[size:64px_64px]" />
      </div>

      {/* Content */}
<div className="relative z-10 max-w-6xl mx-auto text-center transform-none">
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8 }}
        >
          <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-purple-500/10 border border-purple-500/20 backdrop-blur-sm mb-8">
            <Sparkles className="size-4 text-purple-400" />
            <span className="text-purple-300">Next-Generation Intelligence Platform</span>
          </div>
        </motion.div>

        <motion.h1
          className="mb-6 text-5xl md:text-7xl lg:text-7xl font-bold bg-gradient-to-r from-purple-200 via-blue-200 to-purple-200 bg-clip-text text-transparent text-color-white"
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 0.2 }}
        >
          🚀 IntelliStreamAI – Unified OTT + Financial Intelligence Platform
        </motion.h1>

        <motion.p
          className="mb-12 text-slate-300 max-w-4xl mx-auto text-lg md:text-xl leading-relaxed"
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 0.4 }}
        >
          Transform OTT viewer behavior and financial market signals into actionable, AI-driven insights.
          Predict content success, recommend smarter, forecast stocks, and analyze business outcomes — all on one platform.
        </motion.p>
        <div className="flex gap-3 justify-center mb-6">
          {!isAuthenticated ? (
            <>
              <Button
                variant="outline"
                className="transform-none bg-slate-900/60 border-slate-500/30 text-slate-200 hover:bg-slate-800"
                onClick={() => setShowAuthModal("signin")}
              >
                Sign In
              </Button>

              <Button
                className="transform-none bg-purple-600 hover:bg-purple-500 text-white"
                onClick={() => setShowAuthModal("signup")}
              >
                Sign Up
              </Button>
            </>
          ) : (
            <Button
              variant="outline"
              className="transform-none bg-red-500/10 border-red-500/30 text-red-300 hover:bg-red-500/20"
              onClick={() => setIsAuthenticated(false)}
            >
              Logout
            </Button>
          )}
        </div>

        <motion.div
          className="flex flex-wrap gap-4 justify-center"
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 0.6 }}
        >
          <Button
            size="lg"
            className="transform-none bg-gradient-to-r from-purple-600 to-blue-600 hover:from-purple-500 hover:to-blue-500 text-white border-0 shadow-[0_0_30px_rgba(168,85,247,0.4)] transition-all duration-300 hover:shadow-[0_0_40px_rgba(168,85,247,0.6)]"
            onClick={() => {
              const modulesSection = document.getElementById('modules-section');
              modulesSection?.scrollIntoView({ behavior: 'smooth' });
            }}
          >
            <Sparkles className="size-4 mr-2" />
            Start Analysis →
          </Button>
          <Button
            size="lg"
            variant="outline"
            className="transform-none bg-slate-900/50 border-purple-500/30 text-purple-200 hover:bg-purple-500/10 hover:border-purple-500/50 backdrop-blur-sm"
            onClick={() => onNavigate("upload")}
          >
            <Upload className="size-4 mr-2" />
            Upload Your Data →
          </Button>
          <Button
            size="lg"
            variant="outline"
            disabled={!isAuthenticated}
            className={`${!isAuthenticated ? "transform-none pacity-50 cursor-not-allowed" : ""} bg-slate-900/50 border-blue-500/30 text-blue-200 hover:bg-blue-500/10 hover:border-blue-500/50 backdrop-blur-sm`}
            onClick={() => {
              const modulesSection = document.getElementById('modules-section');
              modulesSection?.scrollIntoView({ behavior: 'smooth' });
            }}
          >
            <Layers className="size-4 mr-2" />
            Explore Modules →
          </Button>
        </motion.div>

        {/* Floating Elements */}
        <motion.div
          className="absolute top-1/4 right-1/4 text-6xl opacity-20"
          animate={{
            y: [0, -20, 0],
            rotate: [0, 10, 0],
          }}
          transition={{
            duration: 6,
            repeat: Infinity,
            ease: "easeInOut",
          }}
        >
          📈
        </motion.div>
        <motion.div
          className="absolute bottom-1/3 left-1/4 text-6xl opacity-20"
          animate={{
            y: [0, 20, 0],
            rotate: [0, -10, 0],
          }}
          transition={{
            duration: 7,
            repeat: Infinity,
            ease: "easeInOut",
          }}
        >
          🎬
        </motion.div>
      </div>
    {showAuthModal && (
  <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm">
    <div className="w-full max-w-md rounded-2xl bg-slate-900 border border-slate-700 p-6">
      <h2 className="text-xl font-semibold text-white mb-4">
        {showAuthModal === "signin" ? "Sign In to IntelliStreamAI" : "Create an Account"}
      </h2>

      <form
        className="space-y-4"
        onSubmit={(e) => {
          e.preventDefault();
          setIsAuthenticated(true);
          setShowAuthModal(null);
        }}
      >
        <input
          type="email"
          required
          placeholder="Email"
          className="w-full rounded-lg bg-slate-800 border border-slate-600 px-4 py-2 text-slate-200 focus:outline-none focus:ring-2 focus:ring-purple-500"
        />

        <input
          type="password"
          required
          placeholder="Password"
          className="w-full rounded-lg bg-slate-800 border border-slate-600 px-4 py-2 text-slate-200 focus:outline-none focus:ring-2 focus:ring-purple-500"
        />

        {showAuthModal === "signup" && (
          <input
            type="password"
            required
            placeholder="Confirm Password"
            className="w-full rounded-lg bg-slate-800 border border-slate-600 px-4 py-2 text-slate-200 focus:outline-none focus:ring-2 focus:ring-purple-500"
          />
        )}

        <div className="flex gap-3 justify-end pt-2">
          <Button variant="outline" onClick={() => setShowAuthModal(null)}>
            Cancel
          </Button>
          <Button type="submit">
            {showAuthModal === "signin" ? "Sign In" : "Sign Up"}
          </Button>
        </div>
      </form>
    </div>
  </div>
)}

    </section>
    
  );
}