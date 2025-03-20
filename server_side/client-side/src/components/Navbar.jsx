import { useContext, useState } from "react";
import { DesignsContext } from "../contexts/DesignsContext";
import { ThemeContext } from "../contexts/ThemeContext";



export default function Navbar({ children }){
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
              <svg version="1.0" xmlns="http://www.w3.org/2000/svg" width="1344" height="1344" viewBox="0 0 1008 1008"><path d="M485.9 205.7c0 .5-2.8 21.2-6.2 46.1l-6.2 45.4 1.1 38.1c.6 21 .9 38.6.8 39.2-.2.5-2.4 17.4-4.9 37.5-8.5 67.3-13.4 86.1-32.8 125.5-6.6 13.3-8.1 15.6-8.7 13.9-.5-1.2-2-11.3-3.4-22.5-5.9-45.9-11.1-62.5-25.6-81.7-4.6-6.1-9.2-10.5-19-18.1L368.1 419h-15.5c-15.5 0-15.6 0-15.6 2.2-.1 1.3.9 20.1 2.1 41.8l2.2 39.5-5.6 13c-3 7.1-6.3 14.8-7.2 17l-1.7 4-2.4-11.5c-2.6-12.4-8.1-30.9-12.1-40.5-6.4-15.2-16.7-28.6-27.7-35.7l-5.9-3.8h-25.5l-.4 10.2c-.6 13.8-1.4 15.4-24.7 47.7-7.8 10.9-16.2 26.2-17.7 32.5-.8 3.2-.3 3.2-16.9.1-10.5-1.9-18.8-1.9-25.5.1-22.5 6.5-22.9 6.6-41.5 7.1-11.5.3-26.1-.1-40.4-1.1-12.4-.9-26.3-1.6-31-1.6-7.6 0-8.8.3-11 2.4-3.3 3.1-12.9 6-26.7 8-6 .9-12.4 1.9-14.3 2.2l-3.4.5.7 10.7c.9 15.7.8 15.5 7 14.7 19.6-2.4 35.9-6.1 44.4-10l5.5-2.5 22.9 1.5c42.8 2.8 66.4 1.8 84.3-3.6 14.8-4.4 14.5-4.4 32.9-1 11.5 2.1 19.9 3.1 26.6 3.1h9.8l.3-10.8c.4-10.1.6-11.2 4.5-19.2 2.3-4.7 7.9-13.7 12.5-20 13.4-18.5 19-27.1 22.2-33.9l3-6.4 2.9 3.3c1.5 1.9 4.6 7 6.8 11.5 10.5 21.4 17.5 55.8 18.7 92.7l.6 17.8h7.7c19.7 0 28.2-8.3 31.5-30.9 1.1-7.5 3-14 6.9-23.9 7.5-19.1 13.3-32.2 15.2-34.3 1.5-1.7 1.5-3.9 0-29.1-.9-15-1.6-28.6-1.6-30.1v-2.9l5.2 4.2c8.9 7.1 16.2 18.2 20.8 32.1 3 8.9 5.5 22.1 8.5 45.9 3 23.3 4.5 31.9 8.1 44 2.9 9.5 17.4 40.5 18.4 39.3 1.4-1.6 35.4-65.9 39.6-74.9 17.5-37.9 23-60.4 31.9-132.9 1.9-16 3.8-29.6 4-30.4.3-.7 0-18.5-.6-39.5l-1.2-38.1 2.9-21.3c1.6-12.5 3.1-20.6 3.5-19.5.9 2.5 5.5 31.3 6.8 42.8 5 43.9 7.5 96.2 9.1 190.5 1.2 72.1 2.6 116.6 4.5 148.5.8 13.2 1.5 39.5 1.5 58.5.1 45.8 1.4 59.5 6.9 69.6 3.4 6.4 8.6 9.4 16.2 9.4h5.5l5.6-9.3c9.3-15.5 13.9-27.3 22.8-58.5 2.7-9.5 8.6-29.5 12.9-44.5 15.1-51.4 23.6-85.8 28.5-115.2 3.2-18.9 5.6-37.3 5.6-43 0-7.6 1.8-5.9 5.4 5.2 7.9 24.8 13.5 61 13.6 88v8.3h10c15.6 0 25-3.9 40.1-16.6 15.8-13.4 35.1-38.7 43.4-56.9 2.4-5.2 4-7.5 5.2-7.5 1 0 8.6 6.8 16.8 15 17.3 17.3 16.9 17.1 37.7 19.2 13 1.3 27 .2 56.8-4.8 17-2.8 25.9-2.8 60-.4 15.8 1.2 45.2.9 74.8-.7l12.2-.6v-26.9l-4.2.6c-2.4.3-16.1 1-30.6 1.7-20.8.8-30.9.8-48-.3-35-2.1-55.7-1.7-74.7 1.5-23.9 4.1-33.7 4.9-44.6 3.7l-9.4-1.1-17.2-17.1-17.1-17.1-9.9.6c-18.4 1.1-22.5 4-31.2 22-6.4 13.3-13.8 24.2-25.2 37.3-6.5 7.5-18.5 18.2-19.3 17.3-.2-.2-1.1-6.9-2-14.9-4.8-41.1-15.9-75.6-29-90.6-2.6-2.9-7.1-6.5-10.8-8.4-5.8-3.2-7.1-3.5-17-3.8l-10.8-.4v18.1c0 47.9-9 94.4-37 190.1-4.9 16.8-10.4 35.7-12.2 42l-3.3 11.5-.6-40c-.3-22-.9-42.3-1.3-45-1.3-10-2.6-54.6-4.1-133-1.9-100.2-3.2-133.3-7-178.5-5.4-62.3-13.6-97.1-26.3-111.2-5.5-6.1-12-8.8-21.4-8.8-4.3 0-7.8.3-7.9.7z"/></svg>
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
              <a className="signup-item login">Login</a>
              <a className="signup-item signup">Sign Up</a>
            </div>
          </div>
        </nav>
    </header>
  );
}