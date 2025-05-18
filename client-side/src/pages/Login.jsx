import Section from '../components/Section';
import Form from '../components/Form';  

export default function Login(){
    return (
        <Section section-name={"data-form"}>
            <Form mode="login"/>
        </Section>
    )
}