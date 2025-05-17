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
    event.preventDefault();
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
            <a className="navbar-brand" href="/" onClick={(event) => {
              event.preventDefault();
              document.body.scrollIntoView();
            }}>
              <img src={trlv_logo}/>
            </a>
            
            <div onClick={toggle_menu} className={`button-container ${isOpened === true ? "opened" : ""}`}>
                <a href="#" className="middle-line"></a>
            </div>
          </div>
          
          <div className="nav-menu-container">
            <div className="nav-menu">
              <a className="nav-item about-section" aria-current="page" onClick={close_and_go}>About</a>
              <a className="nav-item store-section" onClick={close_and_go}>Store</a>
              <a className="nav-item joinus-section" onClick={close_and_go}>Join Us</a>
              <a className="nav-item faq-section" onClick={close_and_go}>FAQ</a>
              <a className="nav-item contact-section" onClick={close_and_go}>Contact</a>
            </div>

            <div className="nav-signup">
              <NavLink className="signup-item login" to="/login">Login</NavLink>
              <NavLink className="signup-item signup" to="/signup">Sign Up</NavLink>
            </div>
          </div>
        </nav>
    </header>
  );
}