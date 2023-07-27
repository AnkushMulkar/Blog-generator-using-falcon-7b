import streamlit as st
import os
from langchain import HuggingFaceHub, PromptTemplate, LLMChain
from PIL import Image, ImageDraw, ImageOps


# Set Hugging Face API key
#os.environ['API_KEY'] = 'api_key'

# Initialize the language model
model_id = 'tiiuae/falcon-7b-instruct'
falcon_llm = HuggingFaceHub(huggingfacehub_api_token=os.environ['API_KEY'],
                            repo_id=model_id,
                            model_kwargs={"temperature": 0.8, "max_new_tokens": 2000})

# Define the prompt template
template = """
write a detailed blog with 5 paragraphs about {topic}, that should contain a detailed introduction about {topic}, explanation about {topic} in detail, and summarizes the {topic}.
add Ankush Mulkar at the end.
"""

prompt = PromptTemplate(template=template, input_variables=['topic'])

def create_falcon_chain(llm, prompt):
    return LLMChain(llm=llm, prompt=prompt, verbose=True)

# Create the language model chain
falcon_chain = create_falcon_chain(falcon_llm, prompt)


# Function to crop image in circular shape
def crop_to_circle(image):
    width, height = image.size
    mask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask)
    draw.ellipse((0, 0, width, height), fill=255)
    result = ImageOps.fit(image, mask.size, centering=(0.5, 0.5))
    result.putalpha(mask)
    return result

# Streamlit app
def main():
    st.title("Blog Generator")
    user_input = st.text_input("Text your topic here:")

 # Load the user's profile image
    profile_image = Image.open("a.png")
    profile_image = crop_to_circle(profile_image)

    # Display the profile image in the sidebar
    st.sidebar.image(profile_image, use_column_width=True)

    # Add background music from the current directory
    base_path = os.path.dirname(os.path.abspath(__file__))
    audio_file_path = os.path.join(base_path, "music (2).mp3")
    audio_file = open(audio_file_path, "rb")
    audio_bytes = audio_file.read()
    st.audio(audio_bytes, format="audio/mp3", start_time=0)

    
    if st.button("Generate the Blog"):
        if user_input:
            answer = falcon_chain.run(user_input)
            st.markdown(f"**Blog Generator:** {answer}")
        else:
            st.warning("Please enter a text!")

if __name__ == "__main__":
    main()
