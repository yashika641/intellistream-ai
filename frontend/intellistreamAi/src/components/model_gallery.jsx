import { useRef, useState } from "react";
import { useNavigate } from "react-router-dom";
import { useGSAP } from "@gsap/react";
import gsap from "gsap";

import recommender_model from "../assets/recommender_model.png";
import stock_model from "../assets/stock_model.png";
import churn_model from "../assets/churn_model.png";
import script_model from "../assets/script_success_model.png";

export default function ModelGallery() {
  const navigate = useNavigate();
  const previewRef = useRef(null);
  const [hovered, setHovered] = useState(null);

  const images = [
    { id: 1, src: recommender_model, alt: "Recommender System", link: "/recommender" },
    { id: 2, src: stock_model, alt: "Stock Price Predictor", link: "/stock" },
    { id: 3, src: churn_model, alt: "Customer Churn Predictor", link: "/churn" },
    { id: 4, src: script_model, alt: "Script Success Predictor", link: "/script" },
  ];

  // Smooth fade & scale animation when hovered image changes
  useGSAP(() => {
    if (hovered) {
      gsap.fromTo(
        previewRef.current,
        { opacity: 0, x: -50, scale: 0.9 },
        { opacity: 1, x: 0, scale: 1, duration: 0.6, ease: "power3.out" }
      );
    }
  }, [hovered]);

  return (
    <div className="flex flex-col md:flex-row items-center justify-center w-full min-h-[90vh] text-white overflow-hidden px-10">
      {/* Left Preview Panel */}
      <div className="hidden md:flex w-1/2 justify-center items-center relative">
        {hovered ? (
          <img
            ref={previewRef}
            src={hovered.src}
            alt={hovered.alt}
            className="w-[420px] h-[420px] object-cover rounded-2xl shadow-[0_0_30px_rgba(147,51,234,0.5)] transition-all"
          />
        ) : (
          <div className="text-center my-12">
                <h2 className="text-6xl md:text-5xl font-extrabold mb-4 
                 bg-linear-to-r from-blue-400 via-purple-500 to-pink-500 
                 bg-clip-text text-transparent animate-gradient">
                  Our AI Models
                </h2>
                <div className="h-1 w-40 bg-linear-to-r from-blue-400 via-purple-500 to-pink-500 mx-auto mb-8 rounded-full shadow-lg shadow-purple-500/40"></div>
                <p className="text-gray-300 text-lg max-w-2xl mx-auto">
                  Explore the intelligent core of IntelliStream AI — our advanced models for recommendations, churn prediction, market analysis, and script success forecasting.
                </p>
              </div>
        )}
      </div>

      {/* Right Image List */}
      <div className="flex md:w-1/2 flex-wrap justify-center gap-8">
        {images.map((img) => (
          <div
            key={img.id}
            onMouseEnter={() => setHovered(img)}
            onMouseLeave={() => setHovered(null)}
            onClick={() => navigate(img.link)}
            className="cursor-pointer group transition-all duration-300 hover:scale-110"
          >
            <img
              src={img.src}
              alt={img.alt}
              className="w-64 h-64 object-cover rounded-xl shadow-lg transition-all duration-500 group-hover:shadow-purple-500/50"
            />
            <p className="text-center mt-3 text-gray-300 text-lg group-hover:text-purple-400 transition-colors">
              {img.alt}
            </p>
          </div>
        ))}
      </div>
    </div>
  );
}
