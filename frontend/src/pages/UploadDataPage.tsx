import { Navigation } from "../components/Navigation";
import { Card, CardContent, CardHeader, CardTitle } from "../components/ui/card";
import { motion } from "motion/react";
import { Upload, FileCheck, AlertCircle, CheckCircle, Clock } from "lucide-react";
import { Page } from "../App";
import { Button } from "../components/ui/button";
import { useState } from "react";

interface UploadDataPageProps {
  onNavigate: (page: Page) => void;
}

export function UploadDataPage({ onNavigate }: UploadDataPageProps) {
  const [isDragging, setIsDragging] = useState(false);
  const [uploadedFiles, setUploadedFiles] = useState<Array<{
    name: string;
    size: string;
    status: "processing" | "complete" | "error";
    type: string;
  }>>([]);

  const simulateUpload = () => {
    const newFiles = [
      { name: "user_behavior_data.csv", size: "2.4 MB", status: "processing" as const, type: "CSV" },
      { name: "content_metadata.json", size: "1.8 MB", status: "processing" as const, type: "JSON" },
    ];
    
    setUploadedFiles(newFiles);

    setTimeout(() => {
      setUploadedFiles(prev => prev.map((file, i) => 
        i === 0 ? { ...file, status: "complete" as const } : file
      ));
    }, 2000);

    setTimeout(() => {
      setUploadedFiles(prev => prev.map((file, i) => 
        i === 1 ? { ...file, status: "complete" as const } : file
      ));
    }, 3500);
  };

  return (
    <div className="min-h-screen">
      <Navigation onNavigate={onNavigate} currentPage="Upload Data" />
      
      <div className="max-w-5xl mx-auto px-6 py-12">
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6 }}
        >
          <div className="mb-12">
            <h1 className="mb-4 text-transparent bg-gradient-to-r from-blue-300 to-purple-300 bg-clip-text">
              📁 Upload Your Data
            </h1>
            <p className="text-slate-400 text-lg">
              Upload CSV user logs, script PDFs, or sentiment text files for AI analysis
            </p>
          </div>

          {/* Upload Zone */}
          <Card
            className={`backdrop-blur-lg bg-slate-900/50 border-2 border-dashed transition-all duration-300 cursor-pointer mb-8 ${
              isDragging
                ? "border-purple-500 bg-purple-500/10 shadow-[0_0_40px_rgba(168,85,247,0.4)]"
                : "border-slate-700 hover:border-purple-500/50"
            }`}
            onDragOver={(e) => {
              e.preventDefault();
              setIsDragging(true);
            }}
            onDragLeave={() => setIsDragging(false)}
            onDrop={(e) => {
              e.preventDefault();
              setIsDragging(false);
              simulateUpload();
            }}
            onClick={simulateUpload}
          >
            <CardContent className="p-16">
              <motion.div
                className="text-center"
                animate={{
                  y: isDragging ? -10 : 0,
                  scale: isDragging ? 1.05 : 1,
                }}
                transition={{ duration: 0.3 }}
              >
                <div className="inline-flex items-center justify-center w-24 h-24 rounded-full bg-gradient-to-br from-purple-500/20 to-blue-500/20 border border-purple-500/30 mb-6">
                  <Upload className="size-12 text-purple-400" />
                </div>
                <h3 className="text-white mb-3">
                  {isDragging ? "Drop files here" : "Drag & drop files or click to browse"}
                </h3>
                <p className="text-slate-400 mb-6">
                  Supports CSV, PDF, TXT, JSON, ZIP files up to 100MB
                </p>
                <Button className="bg-gradient-to-r from-purple-600 to-blue-600 hover:from-purple-500 hover:to-blue-500">
                  Select Files
                </Button>
              </motion.div>
            </CardContent>
          </Card>

          {/* File Processing Status */}
          {uploadedFiles.length > 0 && (
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              className="mb-8"
            >
              <Card className="backdrop-blur-lg bg-slate-900/50 border border-blue-500/30">
                <CardHeader>
                  <CardTitle className="text-white">Processing Files</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="space-y-4">
                    {uploadedFiles.map((file, index) => (
                      <div
                        key={index}
                        className="flex items-center justify-between p-4 rounded-lg bg-slate-800/50 border border-slate-700/50"
                      >
                        <div className="flex items-center gap-4 flex-1">
                          <div className={`p-3 rounded-lg ${
                            file.status === 'complete' ? 'bg-green-500/10' :
                            file.status === 'error' ? 'bg-red-500/10' :
                            'bg-blue-500/10'
                          }`}>
                            {file.status === 'complete' ? (
                              <CheckCircle className="size-5 text-green-400" />
                            ) : file.status === 'error' ? (
                              <AlertCircle className="size-5 text-red-400" />
                            ) : (
                              <motion.div
                                animate={{ rotate: 360 }}
                                transition={{ duration: 2, repeat: Infinity, ease: "linear" }}
                              >
                                <Clock className="size-5 text-blue-400" />
                              </motion.div>
                            )}
                          </div>
                          <div className="flex-1">
                            <p className="text-white mb-1">{file.name}</p>
                            <p className="text-slate-400 text-sm">{file.size} • {file.type}</p>
                          </div>
                          <div className={`px-3 py-1 rounded text-sm ${
                            file.status === 'complete' ? 'bg-green-500/20 text-green-400' :
                            file.status === 'error' ? 'bg-red-500/20 text-red-400' :
                            'bg-blue-500/20 text-blue-400'
                          }`}>
                            {file.status === 'complete' ? 'Complete' :
                             file.status === 'error' ? 'Error' :
                             'Processing...'}
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                </CardContent>
              </Card>
            </motion.div>
          )}

          {/* Data Processing Pipeline */}
          <Card className="backdrop-blur-lg bg-slate-900/50 border border-purple-500/30 mb-8">
            <CardHeader>
              <CardTitle className="text-white">Automated Processing Pipeline</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
                {[
                  { step: "1. Upload", desc: "Secure file transfer", icon: Upload },
                  { step: "2. Validate", desc: "Schema detection", icon: FileCheck },
                  { step: "3. Clean", desc: "Auto-preprocessing", icon: CheckCircle },
                  { step: "4. Analyze", desc: "ML model inference", icon: AlertCircle },
                ].map((item, index) => {
                  const Icon = item.icon;
                  return (
                    <div key={index} className="text-center p-4 rounded-lg bg-slate-800/50 border border-slate-700/50">
                      <div className="inline-flex items-center justify-center w-12 h-12 rounded-full bg-purple-500/10 border border-purple-500/30 mb-3">
                        <Icon className="size-5 text-purple-400" />
                      </div>
                      <p className="text-white mb-1">{item.step}</p>
                      <p className="text-slate-400 text-sm">{item.desc}</p>
                    </div>
                  );
                })}
              </div>
            </CardContent>
          </Card>

          {/* Quick Actions */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            <Card 
              className="backdrop-blur-lg bg-slate-900/50 border border-purple-500/30 cursor-pointer hover:scale-105 transition-transform"
              onClick={() => onNavigate("churn")}
            >
              <CardContent className="p-6 text-center">
                <div className="text-4xl mb-3">👥</div>
                <h3 className="text-white mb-2">Analyze Churn</h3>
                <p className="text-slate-400 text-sm">Upload user behavior data</p>
              </CardContent>
            </Card>

            <Card 
              className="backdrop-blur-lg bg-slate-900/50 border border-blue-500/30 cursor-pointer hover:scale-105 transition-transform"
              onClick={() => onNavigate("script")}
            >
              <CardContent className="p-6 text-center">
                <div className="text-4xl mb-3">🎬</div>
                <h3 className="text-white mb-2">Predict Success</h3>
                <p className="text-slate-400 text-sm">Upload script PDFs</p>
              </CardContent>
            </Card>

            <Card 
              className="backdrop-blur-lg bg-slate-900/50 border border-green-500/30 cursor-pointer hover:scale-105 transition-transform"
              onClick={() => onNavigate("stock")}
            >
              <CardContent className="p-6 text-center">
                <div className="text-4xl mb-3">📈</div>
                <h3 className="text-white mb-2">Forecast Stocks</h3>
                <p className="text-slate-400 text-sm">Upload market data</p>
              </CardContent>
            </Card>
          </div>
        </motion.div>
      </div>
    </div>
  );
}
