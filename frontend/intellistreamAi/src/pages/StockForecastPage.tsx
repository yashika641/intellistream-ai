import { Navigation } from "../components/Navigation";
/// <reference types="vite/client" />

import { Card, CardContent, CardHeader, CardTitle } from "../components/ui/card";
import { motion } from "motion/react";
import { TrendingUp, TrendingDown, DollarSign, BarChart3, Activity, Calendar } from "lucide-react";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  CartesianGrid,
  Area,
} from "recharts";
import { Page } from "../App";
import { Button } from "../components/ui/button";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "../components/ui/select";
import { useEffect, useState } from "react";
interface StockForecastPageProps {
  onNavigate: (page: Page) => void;
}


export function StockForecastPage({ onNavigate }: StockForecastPageProps) {
  const [stocks, setStocks] = useState<any[]>([]);
  const [selectedStock, setSelectedStock] = useState("NFLX");
  const [forecastDays, setForecastDays] = useState("30");
  const [model, setModel] = useState("Prophet");
  const [forecastData, setForecastData] = useState<any>(null);

  const fetchStocks = async () => {
  try {
    const symbols = ["NFLX", "DIS", "WBD", "PARA"];
const apiUrl = import.meta.env.VITE_API_URL || "http://localhost:8000";
    const responses = await Promise.all(
      symbols.map(async (symbol) => {
        const res = await fetch(
          `${apiUrl}/api/dashboard/${symbol}?forecast_days=1`
        );

        if (!res.ok) {
          console.error("Failed for", symbol);
          return null;
        }

        return res.json();
      })
    );

    console.log("API responses:", responses);

    const formatted = responses
      .filter(Boolean)
      .filter(data => data?.realtime_data)
      .map(data => ({
        symbol: data.symbol,
        name: data.symbol,
        price: `$${data.realtime_data.current_price}`,
        change: `${data.realtime_data.percent_change}%`,
        trend:
          data.realtime_data.percent_change >= 0 ? "up" : "down"
      }));

    setStocks(formatted);
  } catch (err) {
    console.error("Fetch error:", err);
  }
};


  useEffect(() => {
    fetchStocks();

    const interval = setInterval(() => {
      fetchStocks();
    }, 60000); // 1 minute

    return () => clearInterval(interval);
  }, []);

const apiUrl = import.meta.env.VITE_API_URL || "http://127.0.0.1:8000";
  const fetchForecast = async () => {
    const res = await fetch(
      `${apiUrl}/api/dashboard/${selectedStock}?forecast_days=${forecastDays}`
    );

    const data = await res.json();
    setForecastData(data);
  };
  useEffect(() => {
    fetchForecast();
  }, [selectedStock, forecastDays, model]);
console.log("Stocks state:", stocks);
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
              <Select value={forecastDays} onValueChange={setForecastDays}>
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
                onClick={() => setSelectedStock(stock.symbol)}
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
                  {selectedStock} - {forecastDays} Day Forecast
                </CardTitle>
                <div className="flex gap-2">
                  <Button
  size="sm"
  variant={model === "Prophet" ? "default" : "outline"}
  onClick={() => setModel("Prophet")}
>
  Prophet
</Button>

<Button
  size="sm"
  variant={model === "ARIMA" ? "default" : "outline"}
  onClick={() => setModel("ARIMA")}
>
  ARIMA
</Button>

<Button
  size="sm"
  variant={model === "LSTM" ? "default" : "outline"}
  onClick={() => setModel("LSTM")}
>
  LSTM
</Button>
                </div>
              </div>
            </CardHeader>
            <CardContent>
              <div className="h-96 bg-slate-950/50 border border-slate-800 rounded-lg p-4">
  {forecastData?.forecast ? (
    <ResponsiveContainer width="100%" height="100%">
      <LineChart data={forecastData.forecast}>
        <CartesianGrid stroke="#1e293b" strokeDasharray="3 3" />
        <XAxis
          dataKey="ds"
          tick={{ fill: "#94a3b8", fontSize: 12 }}
          tickFormatter={(value) => value.slice(5, 10)}
        />
        <YAxis tick={{ fill: "#94a3b8", fontSize: 12 }} />
        <Tooltip
          contentStyle={{
            backgroundColor: "#0f172a",
            border: "1px solid #334155",
          }}
          labelStyle={{ color: "#fff" }}
        />

        {/* Confidence Interval Area */}
        <Area
          type="monotone"
          dataKey="yhat_upper"
          stroke="none"
          fill="#3b82f6"
          fillOpacity={0.1}
        />
        <Area
          type="monotone"
          dataKey="yhat_lower"
          stroke="none"
          fill="#3b82f6"
          fillOpacity={0.1}
        />

        {/* Main Forecast Line */}
        <Line
          type="monotone"
          dataKey="yhat"
          stroke="#3b82f6"
          strokeWidth={3}
          dot={false}
        />
      </LineChart>
    </ResponsiveContainer>
  ) : (
    <div className="flex items-center justify-center h-full text-slate-500">
      Loading forecast...
    </div>
  )}
</div>
              <div className="grid grid-cols-4 gap-4 mt-6">
                <div className="p-4 rounded-lg bg-slate-800/50 text-center">
                  <p className="text-slate-400 mb-1">Current Price</p>
                  <p className="text-white">
  ${forecastData?.realtime_data?.current_price}
</p>
                </div>
                <div className="p-4 rounded-lg bg-slate-800/50 text-center">
                  <p className="text-slate-400 mb-1">Predicted (30d)</p>
                  <p className="text-green-400">
  ${forecastData?.forecast?.[forecastData.forecast.length - 1]?.yhat?.toFixed(2)}
</p>
                </div>
                <div className="p-4 rounded-lg bg-slate-800/50 text-center">
                  <p className="text-slate-400 mb-1">Confidence</p>
                  <p className="text-blue-400">89.2%</p>
                </div>
                <div className="p-4 rounded-lg bg-slate-800/50 text-center">
                  <p className="text-slate-400 mb-1">Expected Gain</p>
                  <p className="text-green-400">
  {(
    ((forecastData?.forecast?.[forecastData.forecast.length - 1]?.yhat -
      forecastData?.realtime_data?.current_price)
      / forecastData?.realtime_data?.current_price) *
    100
  ).toFixed(2)}%
</p>
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
