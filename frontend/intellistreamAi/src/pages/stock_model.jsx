import { useRef } from "react";
import { useGSAP } from "@gsap/react";
import gsap from "gsap";

export default function Recommender() {
  const ref = useRef(null);
  useGSAP(() => {
    gsap.fromTo(ref.current, { opacity: 0, y: 50 }, { opacity: 1, y: 0, duration: 0.6 });
  }, []);

  return (
    <div ref={ref} className="h-screen w-full flex items-center justify-center bg-black text-white text-4xl">
      stock Model Page
    </div>
  );
}
