import { ThemeContext } from './contexts/ThemeContext';
import { DesignsProvider } from './contexts/DesignsContext';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';

import './globals.css'
import './App.css';
import './navbar-862-and-up.css';
import './navbar-862-down.css';

import Navbar from './components/Navbar';
import Login from './pages/Login';
import Signup from './pages/Signup';

import Footer from './components/Footer';
import Landing from './components/Landing';
import About from './components/About';
import Store from './components/Store';
import JoinUs from './components/JoinUs';
import FAQ from './components/FAQ';
import Contact from './components/Contact';
import ProtectedRoute from './components/ProtectedRoute';


function App(){

  return (
    <DesignsProvider>
      <ThemeContext.Provider value={{design: "sharp-minimal", theme: "light"}}>
        <Router>
          <Navbar/>
          <Routes>
            <Route path="/" element={
              <Landing/>
            }/>
            <Route path="/about" element={
              <About/>
            }/>
            <Route path="/store" element={
              <ProtectedRoute>
                <Store/>
              </ProtectedRoute>
            }/>
            <Route path="/join-us" element={
              <JoinUs/>
            }/>
            <Route path="/faq" element={
              <FAQ/>
            }/>
            <Route path="/contact" element={
              <Contact/>
            }/>
            <Route path="/login" element={
              <Login/>
            }/>
            <Route path="/signup" element={
              <Signup/>
            }/>
          </Routes>
          <Footer/>
        </Router>
      </ThemeContext.Provider>
    </DesignsProvider> 
  );
}

export default App
