import { useState } from 'react'
import logo1 from './assets/logo1.png'
import avatar1 from './assets/avatar1.png'
import bg_image from './assets/bg_image.png'
import recommender_model from './assets/recommender_model.png'
import stock_model from './assets/stock_model.png'
import churn_model from './assets/churn_model.png'
import script_model from './assets/script_success_model.png'
import viteLogo from '/vite.svg'
import './App.css'
import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import ModelGallery from "./components/model_gallery";
import Recommender from "./pages/recommender";
import Stock from "./pages/stock_model";
import Churn from "./pages/churn_model";
import Script from "./pages/script_model";

function App() {
  const [count, setCount] = useState(0)

  return (
    <div
      className='flex min-h-screen w-100% flex-col items-center justify-center overflow-x-hidden'
    >
      <img src={bg_image} alt="Background" className="fixed top-0 left-0 w-full h-full -z-10 object-cover" />
      <nav className="fixed top-0 w-full h-16 backdrop-blur-xl flex items-center justify-between px-8 z-50">
        <div>
          <img src={logo1} className="h-15" alt="IntelliStream AI Logo" />
        </div>
        <ul className="hidden md:flex gap-6 ">
          <li><a href="#home" className="hover:text-red-600 -2xl">Home</a></li>
          <li><a href="#about" className="hover:text-red-600">About</a></li>
          <li><a href="#services" className="hover:text-red-600">Services</a></li>
          <li><a href="#contact" className="hover:text-red-600">Contact</a></li>
        </ul>
      </nav>
      <main className="flex flex-col items-center justify-center text-white mt-20 mb-10 px-4">
        <div className="flex flex-col items-center w-50% mt-40 justify-center text-center ">
          <h1 className="text-5xl md:text-7xl font-bold mb-4">IntelliStream AI</h1>
          <p className="text-xl md:text-2xl mb-8 max-w-3xl">
            Empowering Businesses with Intelligent AI Solutions for Enhanced Decision-Making and Growth
          </p>
          <img src={avatar1} alt="Avatar" className="w-32 h-32 rounded-full mb-8 shadow-lg" />
        </div>
        <div className="mt-20 w-full mb-10 px-4">
          <h1 className="text-4xl md:text-6xl font-bold mb-6 text-center">Welcome to IntelliStream AI</h1>
          <p className="text-lg md:text-2xl mb-8 text-center max-w-2xl">
            Revolutionizing Business with Cutting-Edge AI Solutions
          </p>
        </div>
      </main>
      <section className='w-screen flex items-center justify-center'>
      <Router>
      <Routes>
        <Route path="/" element={<ModelGallery />} />
        <Route path="/recommender" element={<Recommender />} />
        <Route path="/stock" element={<Stock />} />
        <Route path="/churn" element={<Churn />} />
        <Route path="/script" element={<Script />} />
      </Routes>
    </Router>
    </section>
    <footer className="w-full h-16  flex items-center justify-center text-white mt-10">
      <p>&copy; 2024 IntelliStream AI. All rights reserved.</p>
    </footer> 
    </div>
  )
}

export default App
