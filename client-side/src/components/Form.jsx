import NameInput from "./NameInput";
import CredentialInput from "./CredentialInput";
import MobileNumberInput from './MobileNumberInput';
import CountryCodeInput from './CountryCodeInput';
import MemberRoleInput from './MemberRoleInput';
import BioInput from './BioInput';
import ImageInput from './ImageInput';  
import Button from './Button';

import { useContext, useState } from "react";
import { ThemeContext } from "../contexts/ThemeContext";
import { DesignsContext } from "../contexts/DesignsContext";
import { FormInputsContext } from "../contexts/FormInputsContext";
import api from "../api";



export default function Form({ children, mode }){

    // console.log(mode)
    let [fname, setFname] = useState("");
    let [lname, setLname] = useState("");
    let [mobileNum, setMobileNum] = useState("");
    let [countryCode, setCountryCode] = useState("");
    let [memberRole, setMemberRole] = useState("");
    let [bio, setBio] = useState("");
    let [profileImage, setProfileImage] = useState(null);

    let [email, setEmail] = useState("");
    let [uname, setUname] = useState("");
    let [password, setPassword] = useState("");

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

    // send a post request and retrieve the response and set 
    // the state of the following states for the alert component
    let [response, setResponse] = useState(null);
    let [msgStatus, setMsgStatus] = useState(undefined);
    let [errorType, setErrorType] = useState(null);

    const handleSubmit = async (event) => {
        try {
            event.preventDefault();
            const form_data = new FormData();

            if(mode == "signup"){
                form_data.append('email_address', email);
                form_data.append('first_name', fname);
                form_data.append('last_name', lname);
                form_data.append('mobile_num', mobileNum);
                form_data.append('country_code', countryCode);
                form_data.append('member_role', memberRole);
                form_data.append('bio', bio)
                form_data.append('profile_image', profileImage)
            }
            
            
            form_data.append('user_name', uname);
            form_data.append('password', password);
            

            console.log(uname, password, email, fname, lname, mobileNum, countryCode, memberRole)

            // once data is validated submitted and then extracted
            // reset form components form element
            setFname("");
            setLname("");
            setMobileNum("");
            setCountryCode("+63");
            setMemberRole("Elder");
            setBio("");
            setProfileImage(null);

            setEmail("");
            setUname("");
            setPassword("");

            // send here the data from the contact component to 
            // the backend proxy server. Note this is for development
            // const resp = await api.post("/api/signup", form_data);
            // setResponse(resp);
            
            const url = `http://127.0.0.1:8000/api/${mode}`;
            // for production
            // const url = 'https://project-alexander.vercel.app/send-data';

            const resp = await fetch(url, {
                'method': 'POST',
                'body': form_data,
            });
            setResponse(resp);

            // if response.status is 200 then that means contact information
            // has been successfully sent to the email.js api
            
            if(resp.status === 200){
                setMsgStatus("success");
                console.log(`message has been sent with code ${resp.status}`);

            }else{
                setMsgStatus("failure");
                console.log(`message submission unsucessful. Response status '${resp.status}' occured`);
            }

        }catch(error){
            setMsgStatus("denied");
            setErrorType(error);
            console.log(`Submission denied. Error '${error}' occured`);
        }
    };

    const formInputs = () => {
        let loginInputs = [
            <NameInput name-type="user"/>,
            <CredentialInput cred-type="password"/>
        ]

        if(mode == "signup"){
            // if mode is signup add extra input fields to the variable
            // storing the typical fields for login and signup
            loginInputs.push(
                <CredentialInput cred-type="email"/>,
                <NameInput name-type="first"/>,
                <NameInput name-type="last"/>,
                <MobileNumberInput/>,
                <CountryCodeInput/>,
                <MemberRoleInput/>,
                <BioInput/>,
                <ImageInput/>
            )
        }

        return loginInputs;
    }

    // console.log(`response: ${response}`);
    // console.log(`message status: ${msgStatus}`);
    // console.log(`error type: ${errorType}`);

    return (
        <FormInputsContext.Provider value={{
            fname, setFname, 
            lname, setLname,  
            mobileNum, setMobileNum, 
            countryCode, setCountryCode,
            memberRole, setMemberRole,
            bio, setBio,
            profileImage, setProfileImage,

            email, setEmail,
            uname, setUname,
            password, setPassword,
            
            handleSubmit,
        }}>
            <div className="form-container">
                <form
                    className={`form ${design}`}
                    style={style}
                    method="POST"
                >
                    {formInputs()}
                    <Button/>
                </form>
                <div className={`alert ${msgStatus !== undefined ? 'show' : ''}`} onClick={(event) => {
                    // remove class from alert container to hide it again
                    event.target.classList.remove('show');
                
                    // reset msg_status to undefined in case of another submission
                    setMsgStatus(undefined);
                }}>
                    <div className="alert-wrapper">
                        {msgStatus === "success" || msgStatus === "failed" ? 
                        <span className="alert-message">Message has been sent with code {response?.status}</span> : 
                        <span className="alert-message">Submission denied. Error {errorType?.message} occured</span>}
                    </div>
                </div>
            </div>
        </FormInputsContext.Provider>
    );
}