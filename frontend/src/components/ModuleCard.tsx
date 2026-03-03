import { motion } from "motion/react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "./ui/card";
import { LucideIcon } from "lucide-react";
import { Page } from "../App";

interface ModuleCardProps {
  module: {
    emoji: string;
    title: string;
    description: string;
    icon: LucideIcon;
    color: string;
    page: Page;
  };
  index: number;
  onNavigate: (page: Page) => void;
}

const colorClasses: Record<string, { bg: string; border: string; glow: string }> = {
  purple: {
    bg: "bg-purple-500/10",
    border: "border-purple-500/30",
    glow: "hover:shadow-[0_0_30px_rgba(168,85,247,0.3)]",
  },
  blue: {
    bg: "bg-blue-500/10",
    border: "border-blue-500/30",
    glow: "hover:shadow-[0_0_30px_rgba(59,130,246,0.3)]",
  },
  pink: {
    bg: "bg-pink-500/10",
    border: "border-pink-500/30",
    glow: "hover:shadow-[0_0_30px_rgba(236,72,153,0.3)]",
  },
  green: {
    bg: "bg-green-500/10",
    border: "border-green-500/30",
    glow: "hover:shadow-[0_0_30px_rgba(34,197,94,0.3)]",
  },
  orange: {
    bg: "bg-orange-500/10",
    border: "border-orange-500/30",
    glow: "hover:shadow-[0_0_30px_rgba(249,115,22,0.3)]",
  },
  cyan: {
    bg: "bg-cyan-500/10",
    border: "border-cyan-500/30",
    glow: "hover:shadow-[0_0_30px_rgba(6,182,212,0.3)]",
  },
};

export function ModuleCard({ module, index, onNavigate }: ModuleCardProps) {
  const Icon = module.icon;
  const colors = colorClasses[module.color] || colorClasses.purple;

  return (
    <motion.div
      initial={{ opacity: 0, y: 30 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true }}
      transition={{ duration: 0.5, delay: index * 0.1 }}
    >
      <Card
        className={`group cursor-pointer backdrop-blur-lg bg-slate-900/50 border ${colors.border} ${colors.glow} transition-all duration-300 hover:scale-105 hover:-translate-y-2`}
        onClick={() => onNavigate(module.page)}
      >
        <CardHeader>
          <div className="flex items-start justify-between mb-2">
            <div className="text-5xl">{module.emoji}</div>
            <div className={`p-2 rounded-lg ${colors.bg} transition-all duration-300 group-hover:scale-110`}>
              <Icon className="size-5 text-white" />
            </div>
          </div>
          <CardTitle className="text-white">{module.title}</CardTitle>
        </CardHeader>
        <CardContent>
          <CardDescription className="text-slate-400">
            {module.description}
          </CardDescription>
        </CardContent>
      </Card>
    </motion.div>
  );
}