import streamlit as st
from streamlit_option_menu import option_menu
from PIL import Image
from tensorflow import keras
import numpy as np
from keras.preprocessing.image import load_img

model = keras.models.load_model('PCOSClassification-model.h5')

st.set_page_config(page_title="Polycystic ovary syndrome (PCOS)", page_icon="ðŸ§¬ï¸")

selected = option_menu(
    menu_title=None,
    options=["Description", "Diagnosis", "Precautions & Symptoms"],
    icons=["info-circle", "search", "book"],
    menu_icon="female",
    default_index=0,
    orientation="horizontal"
)

if selected == "Description":
    st.subheader("Overview:")
    st.success("Polycystic ovary syndrome (PCOS) is one of the most common endocrine and metabolic disorders in premenopausal women. Heterogeneous by nature, PCOS is defined by a combination of signs and symptoms of androgen excess and ovarian dysfunction in the absence of other specific diagnoses. The aetiology of this syndrome remains largely unknown, but mounting evidence in the recent literature suggests that PCOS might be a complex multigenic disorder with strong epigenetic and environmental influences, including diet and lifestyle factors.")

    st.subheader("Prevalence:")
    st.success("The World Health Organization (WHO) estimates that a staggering 116 million women globally grapple with PCOS. Statistics report that about 70% of the women suffering from PCOS remain undiagnosed which highlights the substantial prevalence and under-recognition of this condition. In India, as per the Indian Fertility society, the prevalence of PCOS ranges from 3.7% to 22.5%.")

    st.subheader("Impact on Health:")
    st.success("This lifestyle-related ailment leads to a spectrum of metabolic and psychological challenges, including irregular menstrual cycles, hirsutism, sudden weight gain, type 2 diabetes, thyroid irregularities, and increased risk of depression and other psychiatric disorders, significantly affecting overall quality of life.")

    st.subheader("Diagnostic Criteria:")
    st.success("In the last 25 years, several attempts have been made by institution and societies like National Institutes of Health (NIH), European Society of Human Reproduction and Embryology (ESHRE) and American Society for Reproductive Medicine (ASRM) and Androgen Excess Society (AES) to standardize the diagnostic criteria for PCOS. They are based on various combinations of otherwise unexplained hyperandrogenism, anovulation, and the presence of polycystic ovaries observed through ultrasound imaging. This observation is time-consuming, dependent on the sensitivity of the ultrasound equipment, the skill of the operator, the approach (vaginal v/s abdominal) and the weight of the patient.")

elif selected == "Diagnosis":
    st.title("To analyze Ultrasound-scan images")

    uploaded_show_img = st.image([])
    image = st.file_uploader("Upload an Ultrasound scan")
    
    if image is not None: 
        uploaded_show_img.image(image,use_column_width=True)

    button_tumour = st.button("Submit", use_container_width=True)

    if button_tumour:
        if image is None:
            st.warning("Please upload an image before submitting.")
        else:
            image = load_img(image, target_size=(224, 224))
            img = np.array(image)
            img = img / 255.0
            img = img.reshape(1, 224, 224, 3)

            prediction = model.predict(img)
            l={"infected":prediction[0][0],"notinfected":prediction[0][1]}
            def get_key(val):
                for key, value in l.items():
                    if val == value:
                        return key
            
                return "key doesn't exist"
            res=prediction.max()
            
            if get_key(res) == 'infected':
                st.info('PCOS detected')
            else:
                st.info("You are safe")

            # st.image(image,width=300,caption="Uploaded Image")

elif selected == "Precautions & Symptoms":
    st.subheader("Precautions:")
    st.success("While there is no known cure for PCOS, there are ways to manage its symptoms and reduce the risk of complications. Maintaining a healthy lifestyle with regular exercise and a balanced diet can help manage weight and improve symptoms. Managing insulin levels through medication or lifestyle changes may also be recommended. Regular medical check-ups are important for monitoring symptoms and addressing any potential complications.")

    st.subheader("Symptoms:")
    st.success("""The symptoms of PCOS can vary from person to person and may include:\n
    ❄️ Irregular menstrual periods\n
    ❄️ Excess hair growth on the face, chest, and abdomen\n
    ❄️ Weight gain or difficulty losing weight\n
    ❄️ Thinning hair or hair loss from the scalp (alopecia)\n
    ❄️ Skin tags & Acne\n
    ❄️ Pelvic pain, Fertility problems\n
    ❄️ Mood changes, such as depression or anxiety\n\n
Itâ€™s important to see your healthcare provider if youâ€™re experiencing these symptoms.""")
