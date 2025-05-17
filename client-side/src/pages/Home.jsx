import Landing from '../components/Landing';
import About from '../components/About';
import Store from '../components/Store';
import JoinUs from '../components/JoinUs';
import FAQ from '../components/FAQ';
import Contact from '../components/Contact';
import Footer from '../components/Footer';

export default function Home(){
    return (
        <>
            <Landing/>
            <About/>
            <Store/>
            <JoinUs/>
            <FAQ/>
            <Contact/>
            <Footer/>
        </>
    );
}