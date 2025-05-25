import { useContext } from "react";
import { ThemeContext } from "../contexts/ThemeContext";
import { DesignsContext } from "../contexts/DesignsContext";
import { FormInputsContext } from "../contexts/FormInputsContext";

export default function CredentialInput({ children, "cred-type": cred_type }){
    // console.log(cred_type);
    // initialize and define theme of component by using
    // context
    let style;
    const designs = useContext(DesignsContext);
    const themes = useContext(ThemeContext);
    const { design, theme } = themes;
    
    // sometimes themes context will contain only the design 
    // and not the theme key so check if theme key is in themes
    style = designs[design];
    if('theme' in themes){
        style = designs[design][theme];
    }

    // based on the context provider of wrapped Form containing
    // all its states we use the state appropriate to the EmailInput
    // component and its setter to set from this component the state of
    // the form
    let credential, setCredential;
    let { password, setPassword } = useContext(FormInputsContext);
    credential = password, setCredential = setPassword;
    if(cred_type === "email"){
        // if credential type is email then assign credential
        // and setCredential vars to the state and setter
        let { email, setEmail } = useContext(FormInputsContext);
        credential = email, setCredential = setEmail;
    }

    return (
        <div 
            className={`${cred_type}-credential-container ${design}`} 
            style={style}
        >
            <label 
                htmlFor={`${cred_type}-credential-input`}
                className={`${cred_type}-credential-input-label`}
            >{cred_type.charAt(0).toUpperCase() + cred_type.slice(1)}</label>
            <input 
                type={`${cred_type}`}
                name={`${cred_type}_credential_name`}
                id={`${cred_type}-credential-input`}
                className={`${cred_type}-credential-input-field ${design}`} 
                value={credential}
                onChange={(event) => setCredential(event.target.value)}
                placeholder={cred_type == "email" ? "johnmeyer87@gmail.com" : "xxxxxx"}
                required
            />
        </div>
    );
}