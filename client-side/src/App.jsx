import { ThemeContext } from './contexts/ThemeContext';
import { DesignsProvider } from './contexts/DesignsContext';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';

import './App.css';
import './navbar-862-and-up.css';
import './navbar-862-down.css';

import Home from './pages/Home';
import Login from './pages/Login';
import Signup from './pages/Signup';


function App(){

  return (
    <DesignsProvider>
      <ThemeContext.Provider value={{design: "sharp-minimal", theme: 'light'}}>
        <Router>
          <Routes>
            <Route path="/" element={
              <Home/>
            }/>
            <Route path="/login" element={
              <Login/>
            }/>
            <Route path="/signup" element={
              <Signup/>
            }/>
          </Routes>
        </Router>
      </ThemeContext.Provider>
    </DesignsProvider> 
  );
}

export default App
