import { useContext, useEffect, useState } from "react";
import { DesignsContext } from "../contexts/DesignsContext";
import { ThemeContext } from "../contexts/ThemeContext";



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
              <svg version="1.0" xmlns="http://www.w3.org/2000/svg"
                width="2000" height="2000" viewBox="0 0 1008 1008"
                preserveAspectRatio="xMidYMid meet">

                <g transform="translate(0.000000,1008.000000) scale(0.100000,-0.100000)"
                fill="white">
                <path d="M5500 9205 l0 -875 -1780 0 -1780 0 0 -115 0 -115 1780 0 1780 0 0
                -1315 0 -1316 -37 30 c-50 40 -170 94 -274 125 -52 15 -83 29 -81 37 2 6 17
                31 33 56 38 55 73 134 95 215 22 79 25 279 5 358 -89 357 -419 602 -777 577
                -239 -17 -442 -134 -577 -332 -165 -242 -168 -578 -7 -820 22 -33 37 -61 32
                -62 -165 -37 -338 -110 -405 -173 -82 -75 -79 -40 -75 -780 4 -650 4 -656 27
                -740 110 -401 402 -686 791 -774 l96 -21 -461 -720 -462 -720 1039 -3 1038 -2
                0 -860 0 -860 115 0 115 0 0 4050 0 4050 1200 0 1200 0 0 115 0 115 -1200 0
                -1200 0 0 875 0 875 -115 0 -115 0 0 -875z m-855 -2450 c184 -38 361 -176 444
                -345 93 -191 94 -388 2 -575 -46 -94 -141 -211 -197 -242 -172 -99 -275 -128
                -419 -120 -108 7 -189 31 -310 95 -79 41 -96 56 -143 117 -178 237 -209 483
                -91 725 82 167 259 306 439 344 86 19 189 19 275 1z m-554 -1254 c214 -150
                504 -172 744 -56 39 19 95 53 125 76 l54 42 88 -21 c185 -45 330 -122 353
                -186 41 -118 -288 -249 -720 -286 -431 -36 -928 36 -1116 163 -109 73 -115
                120 -24 189 62 47 148 82 290 118 87 23 113 26 126 17 9 -7 45 -32 80 -56z
                m1389 -1526 c-86 -336 -358 -604 -700 -689 -53 -13 -110 -31 -128 -40 -31 -16
                -35 -24 -34 -76 0 -16 160 -274 421 -682 232 -361 421 -659 421 -662 0 -3
                -416 -6 -925 -6 -509 0 -925 3 -925 6 0 3 181 287 401 632 483 753 439 681
                439 715 0 51 -43 79 -171 109 -357 85 -619 337 -720 693 -23 78 -23 93 -27
                634 -2 304 1 551 5 549 188 -94 283 -126 472 -158 387 -65 854 -47 1177 45
                103 30 260 103 289 135 20 22 20 22 23 -551 2 -549 1 -577 -18 -654z m18
                -1575 l-3 -438 -386 601 c-212 331 -385 603 -384 604 1 2 46 14 100 28 261 66
                514 257 642 485 l28 50 3 -446 c1 -245 1 -643 0 -884z"/>
                </g>
              </svg>
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