import { Navigate } from "react-router-dom";
import { jwtDecode } from "jwt-decode";
import { api } from "../api";

import { REFRESH_TOKEN, ACCESS_TOKEN } from "../constants";
import { useEffect, useState } from "react";

// this component basically wraps around another component/page
// and restricts a user from using this component/page if the user
// has no access token present
export default function ProtectedRoute({ children }){
    const [isAuthorized, setisAuthorized] = useState(null);

    // note that once this mounts it will run the asynchronous
    // auth() function which gets our access token if it indeed
    // has been set in our browser local storage through perhaps
    // a login component
    useEffect(() => {
        auth().catch(() => setisAuthorized(true));
    }, []);

    const refreshToken = async () => {
        const refreshed_token = localStorage.getItem(REFRESH_TOKEN);

        try{
            // send request to backend to get a token using
            // our refresh token
            const response = await api.post("/api/token/refresh/", {
                refresh: refreshed_token
            });

            // if the response is successful its status code will
            // be 200
            if(response.status === 200){
                localStorage.setItem(response.data.access);
                setisAuthorized(true);
            }else{
                setisAuthorized(false);
            }
        }catch(error){
            console.log(error);
            setisAuthorized(false);
        }
    }

    const auth = async () => {
        // it is assumed that our access token has already
        // been set into storage since it has been done so
        // by some sort of login component
        const token  = localStorage.getItem(ACCESS_TOKEN);

        if(!token){
            setisAuthorized(false);
            return
        }

        const decoded = jwtDecode(token)
        const tokenExpiration = decoded.exp
        const now = Date.now() / 1000

        if(tokenExpiration < now){
            // this fires if 
            await refreshToken();
        }else{
            // a redundant function that still allows our session
            // to persist so long as our token is still unexpired
            setisAuthorized(true);
        }
    }

    if(isAuthorized === null){
        return <div>Loading...</div>
    }

    return isAuthorized ? children : <Navigate to="/login"/>
}