import { Navigation } from "../components/Navigation";
import { Card, CardContent, CardHeader, CardTitle } from "../components/ui/card";
import { motion } from "motion/react";
import { TrendingUp, TrendingDown, DollarSign, BarChart3, Activity, Calendar } from "lucide-react";
import { Page } from "../App";
import { Button } from "../components/ui/button";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "../components/ui/select";

interface StockForecastPageProps {
  onNavigate: (page: Page) => void;
}

const stocks = [
  { symbol: "NFLX", name: "Netflix", price: "$487.32", change: "+2.4%", trend: "up" },
  { symbol: "DIS", name: "Disney", price: "$102.45", change: "+1.8%", trend: "up" },
  { symbol: "WBD", name: "Warner Bros", price: "$11.23", change: "-0.5%", trend: "down" },
  { symbol: "PARA", name: "Paramount", price: "$14.67", change: "+3.2%", trend: "up" },
];

export function StockForecastPage({ onNavigate }: StockForecastPageProps) {
  return (
    <div className="min-h-screen">
      <Navigation onNavigate={onNavigate} currentPage="Stock Market Trend Forecaster" />
      
      <div className="max-w-7xl mx-auto px-6 py-12">
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6 }}
        >
          <div className="mb-12 flex items-start justify-between">
            <div>
              <h1 className="mb-4 text-transparent bg-gradient-to-r from-green-300 to-blue-300 bg-clip-text">
                📈 Stock Market Trend Forecaster
              </h1>
              <p className="text-slate-400 text-lg">
                Forecast media stock prices using Prophet, ARIMA & LSTM models
              </p>
            </div>
            <div className="flex gap-3">
              <Select defaultValue="30">
                <SelectTrigger className="w-32 bg-slate-800/50 border-slate-700 text-white">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="7">7 Days</SelectItem>
                  <SelectItem value="30">30 Days</SelectItem>
                  <SelectItem value="90">90 Days</SelectItem>
                </SelectContent>
              </Select>
              <Button className="bg-gradient-to-r from-green-600 to-blue-600 hover:from-green-500 hover:to-blue-500">
                <Calendar className="size-4 mr-2" />
                Update Forecast
              </Button>
            </div>
          </div>

          {/* Stock Watchlist */}
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
            {stocks.map((stock, index) => (
              <Card 
                key={index}
                className={`backdrop-blur-lg bg-slate-900/50 border ${stock.trend === 'up' ? 'border-green-500/30' : 'border-red-500/30'} cursor-pointer hover:scale-105 transition-transform`}
              >
                <CardContent className="p-6">
                  <div className="flex items-start justify-between mb-4">
                    <div>
                      <p className="text-slate-400 mb-1">{stock.symbol}</p>
                      <p className="text-white">{stock.price}</p>
                    </div>
                    <div className={`p-3 rounded-lg ${stock.trend === 'up' ? 'bg-green-500/10' : 'bg-red-500/10'}`}>
                      {stock.trend === 'up' ? (
                        <TrendingUp className="size-5 text-green-400" />
                      ) : (
                        <TrendingDown className="size-5 text-red-400" />
                      )}
                    </div>
                  </div>
                  <div className={stock.trend === 'up' ? 'text-green-400' : 'text-red-400'}>
                    {stock.change}
                  </div>
                  <p className="text-slate-500 text-sm mt-2">{stock.name}</p>
                </CardContent>
              </Card>
            ))}
          </div>

          {/* Main Forecast Chart */}
          <Card className="backdrop-blur-lg bg-slate-900/50 border border-blue-500/30 mb-8 shadow-[0_0_30px_rgba(59,130,246,0.2)]">
            <CardHeader>
              <div className="flex items-center justify-between">
                <CardTitle className="text-white flex items-center gap-2">
                  <BarChart3 className="size-5 text-blue-400" />
                  NFLX - 30 Day Forecast
                </CardTitle>
                <div className="flex gap-2">
                  <Button size="sm" variant="outline" className="bg-slate-800/50 border-slate-700 text-slate-300">
                    Prophet
                  </Button>
                  <Button size="sm" variant="outline" className="bg-slate-800/50 border-slate-700 text-slate-300">
                    ARIMA
                  </Button>
                  <Button size="sm" className="bg-blue-600 hover:bg-blue-500">
                    LSTM
                  </Button>
                </div>
              </div>
            </CardHeader>
            <CardContent>
              <div className="h-96 flex items-center justify-center text-slate-500 border border-slate-800 rounded-lg bg-slate-950/50">
                <div className="text-center">
                  <Activity className="size-16 mx-auto mb-4 text-blue-400" />
                  <p className="mb-2">Interactive Stock Price Forecast Chart</p>
                  <p className="text-sm text-slate-600">Prophet + LSTM hybrid model with 95% confidence intervals</p>
                </div>
              </div>
              
              <div className="grid grid-cols-4 gap-4 mt-6">
                <div className="p-4 rounded-lg bg-slate-800/50 text-center">
                  <p className="text-slate-400 mb-1">Current Price</p>
                  <p className="text-white">$487.32</p>
                </div>
                <div className="p-4 rounded-lg bg-slate-800/50 text-center">
                  <p className="text-slate-400 mb-1">Predicted (30d)</p>
                  <p className="text-green-400">$512.45</p>
                </div>
                <div className="p-4 rounded-lg bg-slate-800/50 text-center">
                  <p className="text-slate-400 mb-1">Confidence</p>
                  <p className="text-blue-400">89.2%</p>
                </div>
                <div className="p-4 rounded-lg bg-slate-800/50 text-center">
                  <p className="text-slate-400 mb-1">Expected Gain</p>
                  <p className="text-green-400">+5.15%</p>
                </div>
              </div>
            </CardContent>
          </Card>

          {/* Model Comparison */}
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-8">
            <Card className="backdrop-blur-lg bg-slate-900/50 border border-purple-500/30">
              <CardHeader>
                <CardTitle className="text-white">Prophet Model</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  <div>
                    <div className="flex justify-between text-sm mb-2">
                      <span className="text-slate-400">Accuracy (MAPE)</span>
                      <span className="text-white">4.2%</span>
                    </div>
                    <div className="h-2 bg-slate-800 rounded-full overflow-hidden">
                      <div className="h-full bg-gradient-to-r from-purple-500 to-purple-600" style={{ width: '95.8%' }} />
                    </div>
                  </div>
                  <div>
                    <div className="flex justify-between text-sm mb-2">
                      <span className="text-slate-400">R² Score</span>
                      <span className="text-white">0.87</span>
                    </div>
                    <div className="h-2 bg-slate-800 rounded-full overflow-hidden">
                      <div className="h-full bg-gradient-to-r from-purple-500 to-purple-600" style={{ width: '87%' }} />
                    </div>
                  </div>
                  <p className="text-slate-500 text-sm">Best for: Trend & seasonality</p>
                </div>
              </CardContent>
            </Card>

            <Card className="backdrop-blur-lg bg-slate-900/50 border border-blue-500/30">
              <CardHeader>
                <CardTitle className="text-white">ARIMA Model</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  <div>
                    <div className="flex justify-between text-sm mb-2">
                      <span className="text-slate-400">Accuracy (MAPE)</span>
                      <span className="text-white">5.8%</span>
                    </div>
                    <div className="h-2 bg-slate-800 rounded-full overflow-hidden">
                      <div className="h-full bg-gradient-to-r from-blue-500 to-blue-600" style={{ width: '94.2%' }} />
                    </div>
                  </div>
                  <div>
                    <div className="flex justify-between text-sm mb-2">
                      <span className="text-slate-400">R² Score</span>
                      <span className="text-white">0.82</span>
                    </div>
                    <div className="h-2 bg-slate-800 rounded-full overflow-hidden">
                      <div className="h-full bg-gradient-to-r from-blue-500 to-blue-600" style={{ width: '82%' }} />
                    </div>
                  </div>
                  <p className="text-slate-500 text-sm">Best for: Short-term forecasts</p>
                </div>
              </CardContent>
            </Card>

            <Card className="backdrop-blur-lg bg-slate-900/50 border border-green-500/30">
              <CardHeader>
                <CardTitle className="text-white">LSTM Model</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  <div>
                    <div className="flex justify-between text-sm mb-2">
                      <span className="text-slate-400">Accuracy (MAPE)</span>
                      <span className="text-white">3.1%</span>
                    </div>
                    <div className="h-2 bg-slate-800 rounded-full overflow-hidden">
                      <div className="h-full bg-gradient-to-r from-green-500 to-green-600" style={{ width: '96.9%' }} />
                    </div>
                  </div>
                  <div>
                    <div className="flex justify-between text-sm mb-2">
                      <span className="text-slate-400">R² Score</span>
                      <span className="text-white">0.93</span>
                    </div>
                    <div className="h-2 bg-slate-800 rounded-full overflow-hidden">
                      <div className="h-full bg-gradient-to-r from-green-500 to-green-600" style={{ width: '93%' }} />
                    </div>
                  </div>
                  <p className="text-slate-500 text-sm">Best for: Complex patterns ⭐</p>
                </div>
              </CardContent>
            </Card>
          </div>

          {/* Market Insights */}
          <Card className="backdrop-blur-lg bg-slate-900/50 border border-cyan-500/30">
            <CardHeader>
              <CardTitle className="text-white flex items-center gap-2">
                <DollarSign className="size-5 text-cyan-400" />
                AI Market Insights
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div className="p-4 rounded-lg bg-gradient-to-r from-green-500/10 to-blue-500/10 border border-green-500/30">
                  <p className="text-white mb-2">📊 Bullish Signal Detected</p>
                  <p className="text-slate-400">Strong upward momentum predicted for Netflix (NFLX) based on subscriber growth and content pipeline.</p>
                </div>
                <div className="p-4 rounded-lg bg-gradient-to-r from-orange-500/10 to-red-500/10 border border-orange-500/30">
                  <p className="text-white mb-2">⚠️ Volatility Warning</p>
                  <p className="text-slate-400">Warner Bros Discovery (WBD) showing increased volatility. Recommend cautious position sizing.</p>
                </div>
              </div>
            </CardContent>
          </Card>
        </motion.div>
      </div>
    </div>
  );
}
