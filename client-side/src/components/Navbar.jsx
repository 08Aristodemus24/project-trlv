import { useContext, useEffect, useState } from "react";
import { DesignsContext } from "../contexts/DesignsContext";
import { ThemeContext } from "../contexts/ThemeContext";
import trlv_logo from "../assets/mediafiles/trlv.svg";
import { Navigate, NavLink } from "react-router-dom";


export default function Navbar(){
  // initialize and define theme of component by using
  // context
  let style;
  const designs = useContext(DesignsContext);
  const themes = useContext(ThemeContext);
  const { design, theme } = themes;
  
  // sometimes themes context will contain only the design 
  // and not the theme key so check if theme key is in themes
  if('theme' in themes){
      style = designs[design][theme];
  }else{
      style = designs[design];
  }

  // state to see if navbar is opened if in mobile
  let [isOpened, setIsOpened] = useState(false)
  const body = document.body;
  
  // if div is closed then its class is .closed if opened then .opened
  const toggle_menu = (event) => {
    event.preventDefault();
    if(isOpened === false){
      setIsOpened(!isOpened);
      body.style.overflow = "hidden";

    }else{
      setIsOpened(!isOpened);
      body.style.overflow = "auto";
    }
  };

  // I don't want links in desktop mode to have access to the modal
  // if navbar is opened then only then can it be closed
  // but what if user opens modal and sets the dims to desktop
  // then when a tag is clicked modal will be closed
  const close_and_go = (event) => {
    // event.preventDefault();
    if(isOpened === true){
      setIsOpened(!isOpened);
      body.style.overflow = "auto";
    }

    const section_id = event.target.classList[1];
    const section = document.querySelector(`#${section_id}`);
    section.scrollIntoView({
      block: 'start',
    });
  }

  return (
    <header className={`navbar-container ${isOpened === true ? "opened" : ""} ${design}`} style={style}>
        <nav className="navbar">
          <div className="nav-brand-container">
            <NavLink className="navbar-brand" to="/" onClick={(event) => {
              document.body.scrollIntoView();
            }}>
              <img src={trlv_logo}/>
            </NavLink>
            
            <div onClick={toggle_menu} className={`button-container ${isOpened === true ? "opened" : ""}`}>
                <a href="#" className="middle-line"></a>
            </div>
          </div>
          
          <div className="nav-menu-container">
            <div className="nav-menu">
              <NavLink className="nav-item about-section" aria-current="page" to="/about" onClick={close_and_go}>About</NavLink>
              <NavLink className="nav-item store-section" to="/store" onClick={close_and_go}>Store</NavLink>
              <NavLink className="nav-item joinus-section" to="/join-us" onClick={close_and_go}>Join Us</NavLink>
              <NavLink className="nav-item faq-section" to="/faq" onClick={close_and_go}>FAQ</NavLink>
              <NavLink className="nav-item contact-section" to="/contact" onClick={close_and_go}>Contact</NavLink>
            </div>

            <div className="nav-signup">
              <NavLink className="signup-item login" to="/login" onClick={close_and_go}>Login</NavLink>
              <NavLink className="signup-item signup" to="/signup" onClick={close_and_go}>Sign Up</NavLink>
            </div>
          </div>
        </nav>
    </header>
  );
}