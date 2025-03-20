import { useContext } from "react";
import { DesignsContext } from "../contexts/DesignsContext";
import { ThemeContext } from "../contexts/ThemeContext";



export default function Footer(){
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

  return (
    <footer className={`footer-container ${design}`} style={style}>
      <div className="footer-content">
          <div className="wrapper">
              <p className="footer-credits">Powered by the Heavenly Father.</p>
              <p className="footer-end">
                  2025 © by The Risen Lord's Vineyard. All rights reserved.
              </p>
          </div>
      </div>
    </footer>
  );
}