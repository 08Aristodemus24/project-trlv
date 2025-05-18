import { useContext, useEffect, useRef, useState } from "react";
import { ThemeContext } from "../contexts/ThemeContext";
import { DesignsContext } from "../contexts/DesignsContext";
import { FormInputsContext } from "../contexts/FormInputsContext";

export default function MemberRoleInput({ children }){
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
    // all its states we use the state appropriate to the CountryCodeInput
    // component and its setter to set from this component the state of
    // the form
    let { memberRole, setMemberRole } = useContext(FormInputsContext);
    let memberRoles = ["Elder", "Disciple", "Youth"];
    
    // set member role to default value upon load
    useEffect(() => {
        setMemberRole(memberRoles[0]);
    }, []);

    return (
        <div 
            className={`member-role-container ${design}`}             
            style={style}
        >
            <label 
                htmlFor="member-role" 
                className="member-role-label">Member Role</label>
            <select 
                name="member_role" 
                id="member-role" 
                className={`member-role-field ${design}`} 
                onChange={(event) => setMemberRole(event.target.value)}
                value={memberRole}
            >
                {memberRoles.map((value, index) => {
                    return (
                        <option
                            key={index}
                            value={value} 
                            label={value}
                        >    
                        </option>
                    );
                })}
            </select>
        </div>
    );
}