import { Navigation } from "../components/Navigation";
import { Card, CardContent, CardHeader, CardTitle } from "../components/ui/card";
import { motion } from "motion/react";
import { Newspaper, TrendingUp, MessageCircle, BarChart3, Radio } from "lucide-react";
import { Page } from "../App";

interface SentimentAnalysisPageProps {
  onNavigate: (page: Page) => void;
}

const newsItems = [
  { 
    headline: "Netflix announces record-breaking Q4 subscriber growth",
    sentiment: "positive",
    score: 0.92,
    source: "Bloomberg",
    impact: "High",
  },
  { 
    headline: "Disney+ faces increased competition in streaming market",
    sentiment: "neutral",
    score: 0.45,
    source: "Reuters",
    impact: "Medium",
  },
  { 
    headline: "Warner Bros Discovery struggles with debt management",
    sentiment: "negative",
    score: -0.78,
    source: "CNBC",
    impact: "High",
  },
  { 
    headline: "Paramount+ launches successful international expansion",
    sentiment: "positive",
    score: 0.84,
    source: "Variety",
    impact: "Medium",
  },
];

const socialBuzz = [
  { topic: "#StrangerThings", mentions: "245K", sentiment: "positive", trend: "up" },
  { topic: "#TheLastOfUs", mentions: "189K", sentiment: "positive", trend: "up" },
  { topic: "#NetflixCancels", mentions: "98K", sentiment: "negative", trend: "down" },
  { topic: "#Disney100", mentions: "156K", sentiment: "positive", trend: "up" },
];

export function SentimentAnalysisPage({ onNavigate }: SentimentAnalysisPageProps) {
  return (
    <div className="min-h-screen">
      <Navigation onNavigate={onNavigate} currentPage="Sentiment & Social Pulse" />
      
      <div className="max-w-7xl mx-auto px-6 py-12">
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6 }}
        >
          <div className="mb-12">
            <h1 className="mb-4 text-transparent bg-gradient-to-r from-orange-300 to-pink-300 bg-clip-text">
              📰 Sentiment & Social Pulse (NLP + News)
            </h1>
            <p className="text-slate-400 text-lg">
              Analyze viewer sentiment and news headlines to detect market-moving buzz
            </p>
          </div>

          {/* Overall Sentiment Dashboard */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
            <Card className="backdrop-blur-lg bg-slate-900/50 border border-green-500/30">
              <CardContent className="p-6">
                <div className="text-center">
                  <div className="inline-flex items-center justify-center w-16 h-16 rounded-full bg-green-500/10 mb-4">
                    <TrendingUp className="size-8 text-green-400" />
                  </div>
                  <p className="text-slate-400 mb-2">Positive Sentiment</p>
                  <p className="text-white text-3xl">67.3%</p>
                  <div className="mt-4 h-2 bg-slate-800 rounded-full overflow-hidden">
                    <div className="h-full bg-gradient-to-r from-green-500 to-green-600" style={{ width: '67.3%' }} />
                  </div>
                </div>
              </CardContent>
            </Card>

            <Card className="backdrop-blur-lg bg-slate-900/50 border border-slate-500/30">
              <CardContent className="p-6">
                <div className="text-center">
                  <div className="inline-flex items-center justify-center w-16 h-16 rounded-full bg-slate-500/10 mb-4">
                    <BarChart3 className="size-8 text-slate-400" />
                  </div>
                  <p className="text-slate-400 mb-2">Neutral Sentiment</p>
                  <p className="text-white text-3xl">21.5%</p>
                  <div className="mt-4 h-2 bg-slate-800 rounded-full overflow-hidden">
                    <div className="h-full bg-gradient-to-r from-slate-500 to-slate-600" style={{ width: '21.5%' }} />
                  </div>
                </div>
              </CardContent>
            </Card>

            <Card className="backdrop-blur-lg bg-slate-900/50 border border-red-500/30">
              <CardContent className="p-6">
                <div className="text-center">
                  <div className="inline-flex items-center justify-center w-16 h-16 rounded-full bg-red-500/10 mb-4">
                    <TrendingUp className="size-8 text-red-400 rotate-180" />
                  </div>
                  <p className="text-slate-400 mb-2">Negative Sentiment</p>
                  <p className="text-white text-3xl">11.2%</p>
                  <div className="mt-4 h-2 bg-slate-800 rounded-full overflow-hidden">
                    <div className="h-full bg-gradient-to-r from-red-500 to-red-600" style={{ width: '11.2%' }} />
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>

          {/* News Sentiment Analysis */}
          <Card className="backdrop-blur-lg bg-slate-900/50 border border-purple-500/30 mb-8">
            <CardHeader>
              <CardTitle className="text-white flex items-center gap-2">
                <Newspaper className="size-5 text-purple-400" />
                Latest News Sentiment Analysis
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                {newsItems.map((news, index) => (
                  <motion.div
                    key={index}
                    initial={{ opacity: 0, x: -20 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ duration: 0.5, delay: index * 0.1 }}
                    className={`p-4 rounded-lg border ${
                      news.sentiment === 'positive' 
                        ? 'bg-green-500/5 border-green-500/30' 
                        : news.sentiment === 'negative'
                        ? 'bg-red-500/5 border-red-500/30'
                        : 'bg-slate-500/5 border-slate-500/30'
                    }`}
                  >
                    <div className="flex items-start justify-between gap-4">
                      <div className="flex-1">
                        <p className="text-white mb-2">{news.headline}</p>
                        <div className="flex items-center gap-4 text-sm">
                          <span className="text-slate-400">{news.source}</span>
                          <span className={`px-2 py-1 rounded ${
                            news.sentiment === 'positive' 
                              ? 'bg-green-500/20 text-green-400' 
                              : news.sentiment === 'negative'
                              ? 'bg-red-500/20 text-red-400'
                              : 'bg-slate-500/20 text-slate-400'
                          }`}>
                            {news.sentiment.charAt(0).toUpperCase() + news.sentiment.slice(1)}
                          </span>
                          <span className="text-slate-500">Impact: {news.impact}</span>
                        </div>
                      </div>
                      <div className="text-right">
                        <p className={`text-2xl ${
                          news.score > 0 ? 'text-green-400' : news.score < 0 ? 'text-red-400' : 'text-slate-400'
                        }`}>
                          {news.score > 0 ? '+' : ''}{news.score.toFixed(2)}
                        </p>
                        <p className="text-slate-500 text-sm">Score</p>
                      </div>
                    </div>
                  </motion.div>
                ))}
              </div>
            </CardContent>
          </Card>

          {/* Social Media Buzz */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <Card className="backdrop-blur-lg bg-slate-900/50 border border-blue-500/30">
              <CardHeader>
                <CardTitle className="text-white flex items-center gap-2">
                  <Radio className="size-5 text-blue-400" />
                  Trending Topics
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  {socialBuzz.map((item, index) => (
                    <div key={index} className="flex items-center justify-between p-4 rounded-lg bg-slate-800/50 border border-slate-700/50">
                      <div className="flex items-center gap-3">
                        <div className={`w-2 h-2 rounded-full ${
                          item.sentiment === 'positive' ? 'bg-green-400' : 'bg-red-400'
                        }`} />
                        <div>
                          <p className="text-white">{item.topic}</p>
                          <p className="text-slate-400 text-sm">{item.mentions} mentions</p>
                        </div>
                      </div>
                      <div className={`flex items-center gap-2 ${
                        item.trend === 'up' ? 'text-green-400' : 'text-red-400'
                      }`}>
                        <TrendingUp className={`size-4 ${item.trend === 'down' ? 'rotate-180' : ''}`} />
                        <span className="text-sm">{item.trend === 'up' ? 'Rising' : 'Falling'}</span>
                      </div>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>

            <Card className="backdrop-blur-lg bg-slate-900/50 border border-pink-500/30">
              <CardHeader>
                <CardTitle className="text-white flex items-center gap-2">
                  <MessageCircle className="size-5 text-pink-400" />
                  Sentiment Over Time
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="h-64 flex items-center justify-center border border-slate-800 rounded-lg bg-slate-950/50">
                  <div className="text-center">
                    <BarChart3 className="size-12 mx-auto mb-3 text-pink-400" />
                    <p className="text-slate-500">Time-series sentiment graph</p>
                    <p className="text-slate-600 text-sm">VADER + RoBERTa analysis</p>
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>
        </motion.div>
      </div>
    </div>
  );
}
