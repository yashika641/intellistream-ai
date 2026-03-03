import { motion } from "motion/react";
import { Card } from "./ui/card";
import { Upload, FileCheck, Database, Shield } from "lucide-react";
import { useState } from "react";
import { Page } from "../App";

interface UploadPanelProps {
  onNavigate: (page: Page) => void;
}

export function UploadPanel({ onNavigate }: UploadPanelProps) {
  const [isDragging, setIsDragging] = useState(false);

  return (
    <section className="relative py-20 px-6">
      <div className="max-w-5xl mx-auto">
        <motion.div
          className="text-center mb-12"
          initial={{ opacity: 0, y: 30 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.6 }}
        >
          <h2 className="mb-4 text-transparent bg-gradient-to-r from-blue-300 to-purple-300 bg-clip-text">
            Upload your files to begin analysis
          </h2>
          <p className="text-slate-400">
            Upload CSV user logs, script PDFs, or sentiment text files. IntelliStreamAI automatically detects file type and renders the right workflow.
          </p>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, y: 30 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.6, delay: 0.2 }}
        >
          <Card
            className={`backdrop-blur-lg bg-slate-900/50 border-2 border-dashed transition-all duration-300 cursor-pointer overflow-hidden ${
              isDragging
                ? "border-purple-500 bg-purple-500/10 shadow-[0_0_40px_rgba(168,85,247,0.4)]"
                : "border-slate-700 hover:border-purple-500/50 hover:bg-purple-500/5"
            }`}
            onDragOver={(e) => {
              e.preventDefault();
              setIsDragging(true);
            }}
            onDragLeave={() => setIsDragging(false)}
            onDrop={(e) => {
              e.preventDefault();
              setIsDragging(false);
              onNavigate("upload");
            }}
            onClick={() => onNavigate("upload")}
          >
            <div className="p-16 text-center">
              <motion.div
                animate={{
                  y: isDragging ? -10 : 0,
                  scale: isDragging ? 1.1 : 1,
                }}
                transition={{ duration: 0.3 }}
              >
                <div className="inline-flex items-center justify-center w-20 h-20 rounded-full bg-gradient-to-br from-purple-500/20 to-blue-500/20 border border-purple-500/30 mb-6">
                  <Upload className="size-10 text-purple-400" />
                </div>
              </motion.div>

              <h3 className="mb-3 text-white">
                {isDragging ? "Drop files here" : "Drop CSV, PDF, TXT, ZIP files or click to browse"}
              </h3>
              <p className="text-slate-400 mb-8">Drag and drop your files anywhere on this area</p>

              <div className="grid grid-cols-1 md:grid-cols-3 gap-4 max-w-2xl mx-auto">
                <div className="flex items-center gap-3 p-3 rounded-lg bg-slate-800/50 border border-slate-700/50">
                  <FileCheck className="size-5 text-green-400 flex-shrink-0" />
                  <span className="text-slate-300">Supports multi-file upload</span>
                </div>
                <div className="flex items-center gap-3 p-3 rounded-lg bg-slate-800/50 border border-slate-700/50">
                  <Database className="size-5 text-blue-400 flex-shrink-0" />
                  <span className="text-slate-300">Auto-cleaning & preprocessing</span>
                </div>
                <div className="flex items-center gap-3 p-3 rounded-lg bg-slate-800/50 border border-slate-700/50">
                  <Shield className="size-5 text-purple-400 flex-shrink-0" />
                  <span className="text-slate-300">Schema detection & validation</span>
                </div>
              </div>
            </div>

            {/* Animated Border Glow */}
            {isDragging && (
              <motion.div
                className="absolute inset-0 border-2 border-purple-500 rounded-lg"
                initial={{ opacity: 0 }}
                animate={{ opacity: [0.5, 1, 0.5] }}
                transition={{ duration: 1.5, repeat: Infinity }}
              />
            )}
          </Card>
        </motion.div>
      </div>
    </section>
  );
}