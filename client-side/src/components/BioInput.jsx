import { useContext } from "react";
import { ThemeContext } from "../contexts/ThemeContext";
import { DesignsContext } from "../contexts/DesignsContext";
import { FormInputsContext } from "../contexts/FormInputsContext";

export default function Bio({ children }){
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

    // based on the context provider of wrapped Form containing
    // all its states we use the state appropriate to the BioInput
    // component and its setter to set from this component the state of
    // the form
    let { bio, setBio } = useContext(FormInputsContext);

    return (
        <div 
            className={`bio-container ${design}`}
            style={style}
        >
            <label 
                htmlFor="bio" 
                className="bio-label"
            >Bio</label>
            <textarea 
                id="bio" 
                rows="5" 
                name="bio" 
                className={`bio-field ${design}`} 
                placeholder="Your bio here" 
                onChange={(event) => setBio(event.target.value)} 
                value={bio}
                required
            />
        </div>
    );
}