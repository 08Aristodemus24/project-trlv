import { ThemeContext } from './contexts/ThemeContext';
import { DesignsProvider } from './contexts/DesignsContext';

import Navbar from './components/Navbar';
import Landing from './components/Landing';
import About from './components/About';
import Store from './components/Store';
import JoinUs from './components/JoinUs';
import FAQ from './components/FAQ';
import Contact from './components/Contact';
import Footer from './components/Footer';


import './App.css';
import './navbar-862-and-up.css';
import './navbar-862-down.css';


function App(){

  return (
    <DesignsProvider>
      <ThemeContext.Provider value={{design: "sharp-minimal", theme: 'light'}}>
        <Navbar/>
        <Landing/>
        <About/>
        <Store/>
        <JoinUs/>
        <FAQ/>
        <Contact/>
        <Footer/>
      </ThemeContext.Provider>
    </DesignsProvider> 
  );
}

export default App
