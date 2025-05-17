import axios from "axios";
import { ACCESS_TOKEN } from "./constants";

// sets the base url to be http://localhost:8000 so that when we call axios 
// we don't use http://localhost:8000 over and over in every request we make
const api = axios.create({
    baseURL: import.meta.env.VITE_API_URL
});

api.interceptors.request.use(
    (config) => {
        const token = localStorage.getItem(ACCESS_TOKEN);
        if(token){
            config.headers.Authorization = `Bearer ${token}`;
        }
    },
    (error) => {
        return Promise.reject(error);
    }
);

export default api;