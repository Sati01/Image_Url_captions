

# Run the Below Commands in CLI if you do not have the required libraries,Packages and modules installed
# %pip install requirements.txt
# %python -m nltk.downloader all

#Import all the required libraries
import requests
import nltk
from PIL import Image
from io import BytesIO
import streamlit as st
from bs4 import BeautifulSoup
from transformers import BlipProcessor, BlipForConditionalGeneration
from nltk.sentiment.vader import SentimentIntensityAnalyzer


# Input for image URL
image_url = st.text_input("Enter the URL of the image:")

#Fetch image from URL and return as PIL Image object.
def fetch_image_from_url(image_url):
    try:
        response = requests.get(image_url)
        response.raise_for_status()  # Ensure we got a valid response

        # Check Content-Type header
        content_type = response.headers.get('Content-Type', '')
        if 'image' in content_type:
            # If the content is an image, process it directly
            img = Image.open(BytesIO(response.content))
            return img
        elif 'text/html' in content_type:
            # If the content is HTML, extract the first image URL
            soup = BeautifulSoup(response.text, 'html.parser')
            img_tag = soup.find('img')
            if img_tag and 'src' in img_tag.attrs:
                img_url = img_tag['src']
                # Handle relative URLs
                img_url = requests.compat.urljoin(image_url, img_url)
                return fetch_image_from_url(img_url)  # Recursively fetch image
            else:
                print("No image found in HTML content.")
                return None
        else:
            print("The URL does not point to an image or HTML.")
            return None
    except requests.exceptions.RequestException as e:
        print(f"Request error: {e}")
        return None
    except IOError as e:
        print(f"Image processing error: {e}")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None
    

if image_url:
        image1 = fetch_image_from_url(image_url)
        # Display the image
        if image1:
            st.image(image1, caption='Image from URL', use_column_width=True)


        from transformers import BlipProcessor, BlipForConditionalGeneration
        from nltk.sentiment.vader import SentimentIntensityAnalyzer
        #Function to generate image description
        def get_image_description(image):
            # Load the pre-trained BLIP model and processor
            processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
            model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

            # Process the image and generate caption
            inputs = processor(images=image, return_tensors="pt")
            out = model.generate(**inputs)
            description = processor.decode(out[0], skip_special_tokens=True)
        
            #Return the caption
            return description
        
        
        #print the description
        description = get_image_description(image1)
        st.write("Caption of the Given Image is: "  + description)
        
        # Initialize the SentimentIntensityAnalyzer
        sid = SentimentIntensityAnalyzer()
        
        # Define the sentence to analyze
        sentence = description
        
        # Get the sentiment scores
        sentiment_scores = sid.polarity_scores(sentence)
        
        # Print the sentiment scores (if required)
        #print("Sentiment scores:", sentiment_scores)
        
        # Function to Interpret the sentiment based on compound scores
        def interpret_sentiment(sentiment_scores):
            compound_score = sentiment_scores['compound']
            if compound_score >= 0.05:
                sentiment = 'Positive'
            elif compound_score <= -0.05:
                sentiment = 'Negative'
            else:
                sentiment = 'Neutral'
            return sentiment
        Sentiment = interpret_sentiment(sentiment_scores)

        #Output into streamlit Box
        output_string = st.text_area("The Sentiment of the Caption is", f"{Sentiment}")
        #Print in CLI 
        print(f"The Sentiment of the Caption is: {interpret_sentiment(sentiment_scores)}")
        