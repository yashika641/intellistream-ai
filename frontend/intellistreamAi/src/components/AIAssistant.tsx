import { motion } from "motion/react";
import { Card, CardContent, CardHeader, CardTitle } from "./ui/card";
import { Button } from "./ui/button";
import { Input } from "./ui/input";
import { Send, Sparkles } from "lucide-react";
import { useState } from "react";

const exampleQuestions = [
  { icon: "🧠", text: "Why is churn high in May?" },
  { icon: "🔍", text: "Which OTT genre leads to longer watch-time?" },
  { icon: "📈", text: "Predict Sony's stock trend for next 30 days." },
  { icon: "💬", text: "Summarize sentiment for Netflix's new show." },
];

interface Message {
  type: "user" | "assistant";
  text: string;
}

export function AIAssistant() {
  const [messages, setMessages] = useState<Message[]>([
    {
      type: "assistant",
      text: "Hello! I'm your IntelliStream AI Assistant. Ask me anything about your OTT analytics or financial forecasts.",
    },
  ]);
  const [input, setInput] = useState("");
  const [isTyping, setIsTyping] = useState(false);

  const handleSend = (text?: string) => {
    const messageText = text || input;
    if (!messageText.trim()) return;

    setMessages((prev) => [...prev, { type: "user", text: messageText }]);
    setInput("");
    setIsTyping(true);

    // Simulate AI response
    setTimeout(() => {
      setMessages((prev) => [
        ...prev,
        {
          type: "assistant",
          text: "I'm processing your request using our advanced AI models. In a production environment, this would provide detailed insights based on your data.",
        },
      ]);
      setIsTyping(false);
    }, 2000);
  };

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
          <h2 className="mb-4 text-transparent bg-gradient-to-r from-purple-300 to-pink-300 bg-clip-text">
            Ask the IntelliStream Assistant
          </h2>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, y: 30 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.6, delay: 0.2 }}
        >
          <Card className="backdrop-blur-lg bg-slate-900/50 border border-purple-500/30 shadow-[0_0_40px_rgba(168,85,247,0.2)]">
            <CardHeader>
              <CardTitle className="flex items-center gap-2 text-white">
                <Sparkles className="size-5 text-purple-400" />
                AI Chat Assistant
              </CardTitle>
            </CardHeader>
            <CardContent>
              {/* Chat Messages */}
              <div className="h-96 overflow-y-auto mb-6 space-y-4 px-2">
                {messages.map((message, index) => (
                  <motion.div
                    key={index}
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.3 }}
                    className={`flex ${message.type === "user" ? "justify-end" : "justify-start"}`}
                  >
                    <div
                      className={`max-w-[80%] p-4 rounded-2xl ${
                        message.type === "user"
                          ? "bg-gradient-to-r from-purple-600 to-blue-600 text-white"
                          : "bg-slate-800/80 border border-slate-700 text-slate-200"
                      }`}
                    >
                      {message.text}
                    </div>
                  </motion.div>
                ))}

                {isTyping && (
                  <motion.div
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    className="flex justify-start"
                  >
                    <div className="bg-slate-800/80 border border-slate-700 p-4 rounded-2xl">
                      <div className="flex gap-2">
                        <motion.div
                          className="w-2 h-2 bg-purple-400 rounded-full"
                          animate={{ scale: [1, 1.5, 1] }}
                          transition={{ duration: 0.6, repeat: Infinity, delay: 0 }}
                        />
                        <motion.div
                          className="w-2 h-2 bg-purple-400 rounded-full"
                          animate={{ scale: [1, 1.5, 1] }}
                          transition={{ duration: 0.6, repeat: Infinity, delay: 0.2 }}
                        />
                        <motion.div
                          className="w-2 h-2 bg-purple-400 rounded-full"
                          animate={{ scale: [1, 1.5, 1] }}
                          transition={{ duration: 0.6, repeat: Infinity, delay: 0.4 }}
                        />
                      </div>
                    </div>
                  </motion.div>
                )}
              </div>

              {/* Example Questions */}
              <div className="mb-4 flex flex-wrap gap-2">
                {exampleQuestions.map((question, index) => (
                  <Button
                    key={index}
                    variant="outline"
                    size="sm"
                    className="bg-slate-800/50 border-slate-700 text-slate-300 hover:bg-purple-500/10 hover:border-purple-500/50"
                    onClick={() => handleSend(question.text)}
                  >
                    <span className="mr-2">{question.icon}</span>
                    {question.text}
                  </Button>
                ))}
              </div>

              {/* Input Area */}
              <div className="flex gap-2">
                <Input
                  placeholder="Ask anything about your data..."
                  value={input}
                  onChange={(e) => setInput(e.target.value)}
                  onKeyPress={(e) => e.key === "Enter" && handleSend()}
                  className="bg-slate-800/50 border-slate-700 text-white placeholder:text-slate-500 focus:border-purple-500/50"
                />
                <Button
                  onClick={() => handleSend()}
                  className="bg-gradient-to-r from-purple-600 to-blue-600 hover:from-purple-500 hover:to-blue-500"
                >
                  <Send className="size-4" />
                </Button>
              </div>
            </CardContent>
          </Card>
        </motion.div>
      </div>
    </section>
  );
}
