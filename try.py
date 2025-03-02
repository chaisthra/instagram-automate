"""
Instagram Message Automation with Python
----------------------------------------
This script demonstrates how to automate sending Instagram messages with AI-generated affirmations and images
using Python, Instabot, Selenium, and the Gemini API.
"""

# Required libraries installation (run these commands first):
# pip install instabot selenium webdriver-manager streamlit google-generativeai pillow requests

import os
import glob
import time
import random
import streamlit as st
import requests
from PIL import Image
from io import BytesIO
import base64
import tempfile
import platform
import sys

# Use only one Google AI import approach - the newer one
from google import genai
from google.genai import types
import matplotlib.pyplot as plt

# Configure Gemini API with fallback options
def configure_gemini(api_key):
    genai.configure(api_key=api_key)
    
    # Get available models
    try:
        models = genai.list_models()
        model_names = [model.name for model in models]
        st.info(f"Available Gemini models: {', '.join(model_names)}")
    except Exception as e:
        st.warning(f"Could not list available models: {str(e)}")
    
    # Try to use Gemini 2.0 Flash first (newest model)
    try:
        model = genai.GenerativeModel('gemini-2.0-flash')
        return model, "gemini-2.0-flash"
    except Exception as e:
        st.warning(f"Could not initialize Gemini 2.0 Flash: {str(e)}")
        
        # Fallback to Gemini 1.5 Pro (good for images)
        try:
            model = genai.GenerativeModel('gemini-1.5-pro')
            return model, "gemini-1.5-pro"
        except Exception as e:
            st.warning(f"Could not initialize Gemini 1.5 Pro: {str(e)}")
            
            # Fallback to Gemini 1.5 Flash
            try:
                model = genai.GenerativeModel('gemini-1.5-flash')
                return model, "gemini-1.5-flash"
            except Exception as e:
                st.warning(f"Could not initialize Gemini 1.5 Flash: {str(e)}")
                
                # Final fallback to Gemini Pro
                try:
                    model = genai.GenerativeModel('gemini-pro')
                    return model, "gemini-pro"
                except Exception as e:
                    st.error(f"Could not initialize any Gemini model: {str(e)}")
                    return None, None

# Generate affirmation using Gemini 1.5 Flash
def generate_affirmation(api_key, theme="positivity"):
    import google.generativeai as genai
    
    try:
        # Configure the Gemini API
        genai.configure(api_key=api_key)
        
        # Use Gemini 1.5 Flash for text generation
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        prompt = f"""Generate a short, uplifting affirmation message about {theme}. 
        The message should be positive, motivational, and no longer than 2 sentences.
        Make it personal, as if speaking directly to someone.
        Use simple language without any special characters."""
        
        # Generate content
        response = model.generate_content(prompt)
        
        # Add success message
        st.success("✅ Successfully generated affirmation using Gemini 1.5 Flash")
        
        # Clean the text to avoid typos
        affirmation_text = response.text.strip()
        # Remove any special characters that might cause typing issues
        affirmation_text = ''.join(c for c in affirmation_text if c.isalnum() or c.isspace() or c in '.,!?')
        
        return affirmation_text
    except Exception as e:
        st.warning(f"Error generating affirmation: {str(e)}")
        
        # Fallback to default affirmations if API fails
        default_affirmations = {
            "positivity": "You are filled with unlimited potential. Every day brings new opportunities for growth.",
            "motivation": "You have the strength to overcome any challenge. Keep pushing forward toward your dreams.",
            "success": "Your hard work is creating the success you deserve. Celebrate each step of your journey.",
            "happiness": "Joy lives within you, not in external circumstances. You deserve to be happy right now.",
            "gratitude": "Your life is filled with blessings worth celebrating. Take a moment to appreciate all that you have.",
            "self-love": "You are worthy of love and respect exactly as you are. Embrace your unique qualities.",
            "mindfulness": "This present moment is a gift. Breathe deeply and connect with the peace that's always within you."
        }
        return default_affirmations.get(theme, "You are capable of amazing things. Today is full of possibilities.")

# Generate image using Gemini with fallback options
def generate_image(model, model_name, affirmation):
    # Check if model supports image generation
    supports_image = any(name in model_name for name in ["2.0", "1.5-pro"])
    
    if not supports_image:
        st.info(f"Model {model_name} may not support image generation. Will try anyway or use default image.")
    
    # Extract two key words from the affirmation
    import re
    from collections import Counter
    
    # Define common stop words to filter out
    stop_words = {'a', 'an', 'the', 'and', 'or', 'but', 'is', 'are', 'was', 'were', 
                 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did',
                 'to', 'from', 'in', 'out', 'on', 'off', 'over', 'under', 'again',
                 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why',
                 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other',
                 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so',
                 'than', 'too', 'very', 'can', 'will', 'just', 'should', 'now', 'of',
                 'with', 'for', 'that', 'this', 'these', 'those', 'your', 'you', 'yours'}
    
    # Tokenize and clean the affirmation
    words = re.findall(r'\b\w+\b', affirmation.lower())
    
    # Filter out stop words and count word frequency
    filtered_words = [word for word in words if word not in stop_words and len(word) > 3]
    
    # Get the two most common words, or use defaults if not enough words
    if len(filtered_words) >= 2:
        word_counts = Counter(filtered_words)
        key_words = [word for word, _ in word_counts.most_common(2)]
    elif len(filtered_words) == 1:
        key_words = [filtered_words[0], "positive"]
    else:
        key_words = ["positive", "vibes"]
    
    # Create a string with the key words
    key_words_text = " ".join(key_words)
    
    prompt = f"""Create a beautiful, inspiring image that visually represents these key words: 
    "{key_words_text}"
    
    Include ONLY the text "{key_words_text}" rendered beautifully in the image, with these specifications:
    - The text should be clearly visible and integrated into the design
    - Use an attractive, modern font that matches the mood
    - The text color should contrast well with the background
    - The text size should be proportional to the image
    
    The overall image should be:
    - Colorful, uplifting, and suitable for sharing on social media
    - Include visual elements that reinforce the meaning of the words
    - Make the image visually appealing with bright colors and positive imagery
    - Do not include any watermarks or additional text
    """
    
    try:
        # Configure generation parameters for better image quality
        generation_config = {
            "temperature": 0.9,
            "top_p": 0.95,
            "top_k": 64,
            "max_output_tokens": 2048,
        }
        
        # Set the model to generate images
        safety_settings = [
            {
                "category": "HARM_CATEGORY_HARASSMENT",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"
            },
            {
                "category": "HARM_CATEGORY_HATE_SPEECH",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"
            },
            {
                "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"
            },
            {
                "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"
            }
        ]
        
        response = model.generate_content(
            prompt,
            generation_config=generation_config,
            safety_settings=safety_settings
        )
        
        # Extract and save the image
        if hasattr(response, 'parts') and response.parts:
            for part in response.parts:
                if hasattr(part, 'image') and part.image:
                    # Save image to a temporary file
                    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
                    with open(temp_file.name, 'wb') as f:
                        f.write(part.image.data)
                    return temp_file.name
        
        # If we get here, image generation failed with this model
        st.warning(f"Image generation with {model_name} didn't produce an image. Trying alternative method...")
        
        # Try using a different model specifically for image generation
        try:
            # Try to use Gemini 1.5 Pro specifically for image generation
            image_model = genai.GenerativeModel('gemini-1.5-pro')
            image_response = image_model.generate_content(prompt)
            
            if hasattr(image_response, 'parts') and image_response.parts:
                for part in image_response.parts:
                    if hasattr(part, 'image') and part.image:
                        # Save image to a temporary file
                        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
                        with open(temp_file.name, 'wb') as f:
                            f.write(part.image.data)
                        return temp_file.name
        except:
            # If that fails too, use default image
            pass
            
        # If all image generation attempts fail, use default
        return get_default_image()
    
    except Exception as e:
        st.warning(f"Image generation failed: {str(e)}. Using default image instead.")
        return get_default_image()

# Get a default inspirational image
def get_default_image():
    # Create a list of URLs to free inspirational images
    default_image_urls = [
        "https://images.unsplash.com/photo-1531278659282-4b94b8c1e3d8",  # Sunset
        "https://images.unsplash.com/photo-1470770841072-f978cf4d019e",  # Mountain
        "https://images.unsplash.com/photo-1469474968028-56623f02e42e",  # Nature
        "https://images.unsplash.com/photo-1506477331477-33d5d8b3dc85",  # Beach
        "https://images.unsplash.com/photo-1507525428034-b723cf961d3e"   # Ocean
    ]
    
    # Choose a random image
    image_url = random.choice(default_image_urls)
    
    try:
        # Download the image
        response = requests.get(image_url)
        if response.status_code == 200:
            # Save to a temporary file
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
            with open(temp_file.name, 'wb') as f:
                f.write(response.content)
            return temp_file.name
    except:
        pass
    
    # If all else fails, return None
    return None

# Method 1: Using Instabot (simpler but with limitations)
def automate_instagram_messages_instabot(username, password, recipients, affirmation, image_path=None, log_callback=None, image_url=None):
    from instabot import Bot
    
    def log(message):
        print(message)
        if log_callback:
            log_callback(message)
    
    # Clear old cookie and session files to prevent errors
    cookie_del = glob.glob("config/*cookie.json")
    session_del = glob.glob("config/*session.json")
    
    for f in cookie_del + session_del:
        try:
            os.remove(f)
            log(f"Deleted old session file: {f}")
        except:
            pass
    
    # Initialize the bot with more options
    bot = Bot(
        max_likes_per_day=50,
        max_follows_per_day=50,
        max_unfollows_per_day=50,
        max_comments_per_day=10,
        max_messages_per_day=20,
        max_likes_to_like=100
    )
    
    try:
        # Login to Instagram
        log("Attempting to log in...")
        bot.login(username=username, password=password)
        
        # List of users to message
        users_to_message = recipients.split(',')
        users_to_message = [user.strip() for user in users_to_message]
        
        # Send messages to users
        for user in users_to_message:
            try:
                log(f"Attempting to send message to {user}...")
                
                # Prepare the full message with the image URL if available
                full_message = affirmation
                if image_url:
                    # Keep the message concise with the shortened URL
                    full_message += f"\n\nHere's your affirmation image: {image_url}"
                
                # Send the affirmation message
                if bot.send_message(full_message, user):
                    log(f"Affirmation sent to {user} successfully")
                else:
                    log(f"Failed to send affirmation to {user}")
                
                # If we have an image, upload and send it
                if image_path and os.path.exists(image_path):
                    # Note: Instabot has limitations with sending images in DMs
                    # This might not work reliably, which is why Selenium is recommended
                    log("Attempting to send image...")
                    if bot.send_photo(image_path, user, caption=""):
                        log(f"Image sent to {user} successfully")
                    else:
                        log(f"Failed to send image to {user}")
                
            except Exception as e:
                log(f"Error sending message to {user}: {str(e)}")
            
            # Add longer delay between messages to avoid being flagged as spam
            delay = random.randint(60, 120)  # 1-2 minute delay
            log(f"Waiting {delay} seconds before sending next message...")
            time.sleep(delay)
        
        # Logout
        bot.logout()
        return True, "Messages sent successfully!"
        
    except Exception as e:
        error_msg = f"Error with Instabot: {str(e)}"
        log(error_msg)
        return False, error_msg

# Method 2: Using Selenium (more robust but more complex)
def automate_instagram_messages_selenium(username, password, recipients, affirmation, image_path=None, log_callback=None, image_url=None):
    from selenium import webdriver
    from selenium.webdriver.common.by import By
    from selenium.webdriver.common.keys import Keys
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    
    def log(message):
        print(message)
        if log_callback:
            log_callback(message)
    
    log("Starting Instagram automation...")
    
    # Initialize the driver
    driver = None
    
    log("Setting up Selenium webdriver...")
    
    # Try Safari first, then Chrome
    try:
        # Try Safari first (native on Mac)
        try:
            log("Attempting to use Safari browser...")
            driver = webdriver.Safari()
            log("Successfully initialized Safari browser")
        except Exception as safari_error:
            log(f"Safari initialization failed: {str(safari_error)}")
            log("Trying Chrome instead...")
            
            # If Safari fails, try Chrome
            try:
                from selenium.webdriver.chrome.service import Service
                from webdriver_manager.chrome import ChromeDriverManager
                
                log("Setting up Chrome browser...")
                options = webdriver.ChromeOptions()
                options.add_argument("--no-sandbox")
                options.add_argument("--disable-dev-shm-usage")
                options.add_argument("--disable-notifications")
                
                service = Service(ChromeDriverManager().install())
                driver = webdriver.Chrome(service=service, options=options)
                log("Successfully initialized Chrome browser")
            except Exception as chrome_error:
                log(f"Chrome initialization failed: {str(chrome_error)}")
                raise Exception("Both Safari and Chrome initialization failed")
    except Exception as e:
        error_msg = f"Failed to initialize any browser: {str(e)}"
        log(error_msg)
        return False, error_msg
    
    if not driver:
        return False, "Could not initialize any browser"
    
    # List of users to message
    users_to_message = recipients.split(',')
    users_to_message = [user.strip() for user in users_to_message]
    
    # Create a shareable URL for the image if it exists
    image_message = ""
    if image_path and os.path.exists(image_path):
        try:
            log("Creating shareable URL for the image...")
            
            # If we have a GCS URL, use it
            if image_url:
                image_message = f"\n\nHere's your affirmation image: {image_url}"
                log("Using shortened GCS image URL")
            else:
                # Fallback to placeholder message
                image_message = "\n\n[Image was generated with Gemini AI - check your app to view it]"
                log("Using placeholder image message")
        except Exception as url_error:
            log(f"Could not create image URL: {str(url_error)}")
    
    try:
        # Open Instagram
        log("Opening Instagram...")
        driver.get("https://www.instagram.com/")
        time.sleep(5)  # Wait for page to load
        
        # Login to Instagram
        log("Logging in to Instagram...")
        
        # Wait for the login form to appear
        WebDriverWait(driver, 20).until(
            EC.presence_of_element_located((By.NAME, "username"))
        )
        
        # Find username and password fields and enter credentials
        username_field = driver.find_element(By.NAME, "username")
        password_field = driver.find_element(By.NAME, "password")
        
        # Clear fields first (important!)
        username_field.clear()
        password_field.clear()
        
        # Type slowly to mimic human behavior
        for char in username:
            username_field.send_keys(char)
            time.sleep(0.1)
            
        for char in password:
            password_field.send_keys(char)
            time.sleep(0.1)
        
        # Click login button
        log("Submitting login form...")
        login_button = driver.find_element(By.XPATH, "//button[@type='submit']")
        login_button.click()
        
        # Wait for login to complete
        time.sleep(10)  # Increased wait time
        
        # Handle all possible dialogs that appear after login
        dialogs_handled = 0
        max_dialogs = 3  # Handle up to 3 dialogs
        
        while dialogs_handled < max_dialogs:
            try:
                # Look for any button containing "Not Now", "Skip", "Cancel", etc.
                dialog_buttons = driver.find_elements(By.XPATH, 
                    "//button[contains(text(), 'Not Now') or contains(text(), 'Skip') or contains(text(), 'Cancel')]")
                
                if dialog_buttons and len(dialog_buttons) > 0:
                    dialog_buttons[0].click()
                    log(f"Clicked dialog button: {dialog_buttons[0].text}")
                    time.sleep(2)
                    dialogs_handled += 1
                else:
                    # No more dialogs found
                    break
            except:
                # No more dialogs or error finding them
                break
        
        # Process each user
        for i, user in enumerate(users_to_message):
            user = user.strip()
            if not user:
                continue
                
            try:
                log(f"Sending message to {user}...")
                
                # Check if this user has a special display name
                display_name = handle_special_accounts(user)
                if display_name:
                    log(f"Found special account: {user} with display name: {display_name}")
                
                # Go directly to the user's profile
                log(f"Navigating to {user}'s profile...")
                driver.get(f"https://www.instagram.com/{user}/")
                time.sleep(5)
                
                # Look for the message button
                try:
                    # Try different selectors for the message button
                    message_button_selectors = [
                        "//button[contains(text(), 'Message')]",
                        "//div[contains(text(), 'Message')]",
                        "//a[contains(@href, '/direct/t/')]"
                    ]
                    
                    message_button = None
                    for selector in message_button_selectors:
                        try:
                            message_button = WebDriverWait(driver, 5).until(
                                EC.element_to_be_clickable((By.XPATH, selector))
                            )
                            if message_button:
                                break
                        except:
                            continue
                    
                    if message_button:
                        message_button.click()
                        log("Clicked message button")
                        time.sleep(5)
                    else:
                        # If we can't find the message button, try direct navigation to DM
                        log("Message button not found, trying direct navigation to DM")
                        driver.get(f"https://www.instagram.com/direct/t/{user}/")
                        time.sleep(5)
                
                except Exception as profile_error:
                    log(f"Error on profile page: {str(profile_error)}")
                    # Try direct navigation to DM as fallback
                    driver.get(f"https://www.instagram.com/direct/t/{user}/")
                    time.sleep(5)
                
                # Verify we're in a DM conversation
                def verify_dm_conversation():
                    try:
                        # Check if we're in a DM conversation by looking for common elements
                        dm_indicators = [
                            "//div[contains(@aria-label, 'Conversation')]",
                            "//div[contains(@role, 'listbox')]",
                            "//div[contains(@aria-label, 'Message')]",
                            f"//h1[contains(text(), '{user}')]",
                            f"//span[contains(text(), '{user}')]"
                        ]
                        
                        # Add indicators for display name if available
                        if display_name:
                            dm_indicators.extend([
                                f"//h1[contains(text(), '{display_name}')]",
                                f"//span[contains(text(), '{display_name}')]"
                            ])
                        
                        for indicator in dm_indicators:
                            try:
                                element = driver.find_element(By.XPATH, indicator)
                                if element:
                                    log(f"DM conversation verified with indicator: {indicator}")
                                    return True
                            except:
                                continue
                        
                        # Take a screenshot for debugging
                        screenshot_path = f"dm_verification_{user}.png"
                        driver.save_screenshot(screenshot_path)
                        log(f"DM verification failed, saved screenshot to {screenshot_path}")
                        return False
                    except Exception as e:
                        log(f"Error verifying DM conversation: {str(e)}")
                        return False
                
                # Check if we're in a DM conversation
                if not verify_dm_conversation():
                    log("Not in DM conversation, trying alternative navigation methods")
                    
                    # Try multiple navigation methods
                    navigation_methods = [
                        # Method 1: Direct URL with username
                        lambda: driver.get(f"https://www.instagram.com/direct/t/{user}/"),
                        
                        # Method 2: Go to main DM page and search for user
                        lambda: (
                            driver.get("https://www.instagram.com/direct/inbox/"),
                            time.sleep(3),
                            # Try to find and click on new message button
                            driver.find_element(By.XPATH, "//button[contains(@aria-label, 'New message')]").click() 
                            if len(driver.find_elements(By.XPATH, "//button[contains(@aria-label, 'New message')]")) > 0 
                            else None,
                            time.sleep(2),
                            # Try to find search input
                            driver.find_element(By.XPATH, "//input[contains(@placeholder, 'Search')]").send_keys(user)
                            if len(driver.find_elements(By.XPATH, "//input[contains(@placeholder, 'Search')]")) > 0
                            else None,
                            time.sleep(2),
                            # Try to click on the user in search results
                            driver.find_element(By.XPATH, f"//div[contains(text(), '{user}')]").click()
                            if len(driver.find_elements(By.XPATH, f"//div[contains(text(), '{user}')]")) > 0
                            else None,
                            time.sleep(3)
                        ),
                        
                        # Method 3: Go to inbox and try to find the user in existing conversations
                        lambda: (
                            driver.get("https://www.instagram.com/direct/inbox/"),
                            time.sleep(5),
                            log("Searching for user in existing conversations..."),
                            # Try to find and click on the user in the conversation list
                            # Use multiple selectors to find the user in the conversation list
                            find_and_click_user_in_conversations(driver, user, log_callback, display_name)
                        )
                    ]
                    
                    for i, method in enumerate(navigation_methods):
                        try:
                            log(f"Trying navigation method {i+1}...")
                            method()
                            time.sleep(5)
                            if verify_dm_conversation():
                                log(f"Navigation method {i+1} successful")
                                break
                        except Exception as nav_error:
                            log(f"Navigation method {i+1} failed: {str(nav_error)}")
                    
                    # Final check
                    if not verify_dm_conversation():
                        log(f"Could not navigate to DM conversation with {user}, skipping")
                        continue  # Skip to next user
                
                # Now we should be in the chat with the user
                # Find the message input field and send the affirmation
                try:
                    # Wait a bit longer for the DM interface to fully load
                    time.sleep(8)
                    
                    # Make sure we've selected the correct conversation
                    try:
                        log("Ensuring the correct conversation is selected...")
                        ensure_conversation_selected(driver, user, log_callback)
                    except Exception as select_error:
                        log(f"Error ensuring conversation is selected: {str(select_error)}")
                    
                    # Try to click on the message area first to activate it
                    try:
                        log("Trying to click on the message area to activate it...")
                        message_area_selectors = [
                            "//div[contains(@aria-label, 'Message')]",
                            "//div[contains(@class, 'message-area')]",
                            "//div[contains(@class, 'chat')]",
                            "//div[contains(@class, 'conversation')]",
                            "//div[contains(@role, 'region')]",
                            "//section[contains(@class, 'message')]",
                            "//div[contains(@class, 'message-thread')]"
                        ]
                        
                        for area_selector in message_area_selectors:
                            try:
                                message_area = driver.find_element(By.XPATH, area_selector)
                                if message_area:
                                    log(f"Found message area with selector: {area_selector}")
                                    message_area.click()
                                    log("Clicked on message area")
                                    time.sleep(2)  # Wait for input to appear after clicking
                                    break
                            except:
                                continue
                    except Exception as area_error:
                        log(f"Error clicking on message area: {str(area_error)}")
                    
                    # Try multiple selectors for the message input
                    message_input_selectors = [
                        "//textarea[@placeholder='Message...']",
                        "//div[@role='textbox']", 
                        "//div[@contenteditable='true']",
                        "//div[contains(@aria-label, 'Message')]",
                        "//textarea[contains(@placeholder, 'Message')]",
                        "//textarea",  # Try any textarea as last resort
                        "//div[@data-testid='message-composer']",  # New Instagram UI
                        "//div[contains(@class, 'focus-visible')]",  # Class-based selector
                        "//div[contains(@class, 'message-composer')]",  # Class-based selector
                        "//div[contains(@class, 'text-input')]",  # Class-based selector
                        "//div[contains(@class, 'composer')]",  # Generic composer class
                        "//div[contains(@class, 'input')]",  # Generic input class
                        "//div[contains(@class, 'editor')]",  # Editor class
                        "//div[contains(@class, 'editable')]",  # Editable content
                        "//div[@role='combobox']",  # Sometimes used for input fields
                        "//div[@role='input']"  # Sometimes used for input fields
                    ]
                    
                    message_input = None
                    for selector in message_input_selectors:
                        try:
                            log(f"Trying to find message input with selector: {selector}")
                            message_input = WebDriverWait(driver, 8).until(  # Increased wait time
                                EC.presence_of_element_located((By.XPATH, selector))
                            )
                            if message_input:
                                log(f"Found message input with selector: {selector}")
                                break
                        except Exception as selector_error:
                            log(f"Selector {selector} failed: {str(selector_error)}")
                            continue
                    
                    if not message_input:
                        # Take a screenshot to debug
                        screenshot_path = f"debug_screenshot_{user}.png"
                        driver.save_screenshot(screenshot_path)
                        log(f"Saved debug screenshot to {screenshot_path}")
                        
                        # Try JavaScript injection as a last resort
                        log("Trying JavaScript injection to find message input...")
                        try:
                            # Try to find any input or contenteditable element that might be the message field
                            js_result = driver.execute_script("""
                                // Try to find message input elements
                                var possibleInputs = [
                                    document.querySelector('textarea[placeholder*="Message"]'),
                                    document.querySelector('div[role="textbox"]'),
                                    document.querySelector('div[contenteditable="true"]'),
                                    document.querySelector('textarea'),
                                    document.querySelector('div[data-testid="message-composer"]'),
                                    document.querySelector('div.focus-visible'),
                                    document.querySelector('div.message-composer'),
                                    document.querySelector('div.text-input'),
                                    document.querySelector('div.composer'),
                                    document.querySelector('div.input'),
                                    document.querySelector('div.editor'),
                                    document.querySelector('div.editable'),
                                    document.querySelector('div[role="combobox"]'),
                                    document.querySelector('div[role="input"]'),
                                    // Try to find by class name containing these terms
                                    document.querySelector('div[class*="composer"]'),
                                    document.querySelector('div[class*="input"]'),
                                    document.querySelector('div[class*="message"]'),
                                    document.querySelector('div[class*="text"]'),
                                    document.querySelector('div[class*="editor"]')
                                ];
                                
                                // Return the first non-null element
                                for (var i = 0; i < possibleInputs.length; i++) {
                                    if (possibleInputs[i]) {
                                        possibleInputs[i].focus();
                                        return true;
                                    }
                                }
                                
                                // If we can't find by querySelector, try a more aggressive approach
                                // Look for any element that might be an input
                                var allDivs = document.querySelectorAll('div');
                                for (var i = 0; i < allDivs.length; i++) {
                                    var div = allDivs[i];
                                    // Check if this div has characteristics of an input field
                                    if (div.getAttribute('contenteditable') === 'true' || 
                                        div.getAttribute('role') === 'textbox' ||
                                        div.getAttribute('role') === 'combobox' ||
                                        div.getAttribute('role') === 'input' ||
                                        (div.className && (
                                            div.className.includes('input') || 
                                            div.className.includes('composer') || 
                                            div.className.includes('editor') || 
                                            div.className.includes('message')
                                        ))) {
                                        div.focus();
                                        return true;
                                    }
                                }
                                
                                return false;
                            """)
                            
                            if js_result:
                                log("Found and focused message input via JavaScript")
                                # Now we can try to send keys directly to the active element
                                active_element = driver.switch_to.active_element
                                message_input = active_element
                            else:
                                raise Exception("Could not find message input field")
                        except Exception as js_error:
                            log(f"JavaScript injection failed: {str(js_error)}")
                            raise Exception("Could not find message input field")
                        
                        # Use clipboard to paste the message instead of typing
                        import pyperclip
                        
                        # Combine affirmation and image URL
                        full_message = affirmation
                        if image_message:
                            full_message += image_message
                        
                        # Check if message is too long and split if necessary
                        if len(full_message) > 500:  # Instagram might have issues with very long messages
                            log("Message is long, splitting into parts")
                            message_parts = [affirmation]
                            if image_message:
                                message_parts.append(image_message.strip())
                        else:
                            message_parts = [full_message]
                        
                        # Send each part of the message
                        for part_index, message_part in enumerate(message_parts):
                            log(f"Sending message part {part_index+1}/{len(message_parts)}")
                            
                            # Copy the message part to clipboard
                            pyperclip.copy(message_part)
                            log(f"Copied message part {part_index+1} to clipboard")
                            
                            # Click the message input to focus it
                            message_input.click()
                            time.sleep(1)
                            
                            # Try direct input first
                            try:
                                log("Trying direct input...")
                                message_input.send_keys(message_part)
                                log("Direct input successful")
                            except Exception as direct_input_error:
                                log(f"Direct input failed: {str(direct_input_error)}")
                                
                                # Fall back to clipboard paste
                                log("Falling back to clipboard paste...")
                                try:
                                    # Paste using keyboard shortcut
                                    if platform.system() == 'Darwin':  # macOS
                                        message_input.send_keys(Keys.COMMAND, 'v')
                                    else:  # Windows/Linux
                                        message_input.send_keys(Keys.CONTROL, 'v')
                                    
                                    log("Pasted message part")
                                except Exception as paste_error:
                                    log(f"Clipboard paste failed: {str(paste_error)}")
                                    
                                    # Last resort: JavaScript injection
                                    log("Trying JavaScript to set input value...")
                                    driver.execute_script("arguments[0].value = arguments[1];", message_input, message_part)
                                    log("Set message via JavaScript")
                            
                            time.sleep(1)
                            
                            # Try multiple methods to send the message
                            send_methods = [
                                # Method 1: Press Enter key
                                lambda: message_input.send_keys(Keys.RETURN),
                                
                                # Method 2: Look for a send button
                                lambda: WebDriverWait(driver, 5).until(
                                    EC.element_to_be_clickable((By.XPATH, "//button[contains(@aria-label, 'Send') or contains(text(), 'Send')]"))
                                ).click(),
                                
                                # Method 3: JavaScript to trigger Enter key
                                lambda: driver.execute_script("""
                                    var element = arguments[0];
                                    var event = new KeyboardEvent('keydown', {
                                        'key': 'Enter',
                                        'code': 'Enter',
                                        'keyCode': 13,
                                        'which': 13,
                                        'bubbles': true
                                    });
                                    element.dispatchEvent(event);
                                """, message_input)
                            ]
                            
                            message_sent = False
                            for i, send_method in enumerate(send_methods):
                                try:
                                    log(f"Trying send method {i+1}...")
                                    send_method()
                                    log(f"Send method {i+1} executed")
                                    time.sleep(3)
                                    
                                    # For simplicity, assume the message was sent if no error occurred
                                    message_sent = True
                                    break
                                except Exception as method_error:
                                    log(f"Send method {i+1} failed: {str(method_error)}")
                            
                            if message_sent:
                                log(f"Successfully sent message part {part_index+1}")
                            else:
                                log(f"Could not send message part {part_index+1}")
                                
                            # Wait between message parts
                            if part_index < len(message_parts) - 1:
                                time.sleep(2)
                        
                        # Final success message
                        if all(message_sent for _ in message_parts):
                            log(f"Successfully sent all message parts to {user}")
                        else:
                            log(f"Some message parts may not have been sent to {user}")
                except Exception as msg_error:
                    log(f"Error sending message: {str(msg_error)}")
                    raise msg_error
                
            except Exception as user_error:
                log(f"Error sending message to {user}: {str(user_error)}")
                continue  # Continue with next user even if this one fails
            
            # Add a delay between messages to avoid rate limiting
            if i < len(users_to_message) - 1:  # If not the last user
                delay = random.randint(5, 10)
                log(f"Waiting {delay} seconds before next message...")
                time.sleep(delay)
        
        log("All messages sent successfully!")
        return True, "Messages sent successfully to all recipients!"
        
    except Exception as e:
        error_msg = f"Error during message sending: {str(e)}"
        log(error_msg)
        return False, error_msg
    
    finally:
        # Close the browser
        log("Closing browser...")
        try:
            driver.quit()
        except:
            pass

# Update the image generation function to upload to GCS
def generate_image_with_imagen(api_key, affirmation, log_callback=None):
    def log(message):
        print(message)
        if log_callback:
            log_callback(message)
    
    try:
        # Extract two key words from the affirmation
        import re
        from collections import Counter
        
        # Define common stop words to filter out
        stop_words = {'a', 'an', 'the', 'and', 'or', 'but', 'is', 'are', 'was', 'were', 
                     'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did',
                     'to', 'from', 'in', 'out', 'on', 'off', 'over', 'under', 'again',
                     'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why',
                     'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other',
                     'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so',
                     'than', 'too', 'very', 'can', 'will', 'just', 'should', 'now', 'of',
                     'with', 'for', 'that', 'this', 'these', 'those', 'your', 'you', 'yours'}
        
        # Tokenize and clean the affirmation
        words = re.findall(r'\b\w+\b', affirmation.lower())
        
        # Filter out stop words and count word frequency
        filtered_words = [word for word in words if word not in stop_words and len(word) > 3]
        
        # Get the two most common words, or use defaults if not enough words
        if len(filtered_words) >= 2:
            word_counts = Counter(filtered_words)
            key_words = [word for word, _ in word_counts.most_common(2)]
        elif len(filtered_words) == 1:
            key_words = [filtered_words[0], "positive"]
        else:
            key_words = ["positive", "vibes"]
        
        # Create a string with the key words
        key_words_text = " ".join(key_words)
        log(f"Extracted key words for image: '{key_words_text}'")
        
        # First try with Gemini for text-to-image prompt enhancement
        log("Using Gemini to create an enhanced image prompt...")
        import google.generativeai as genai
        import json
        import subprocess
        import base64
        import tempfile
        import os
        
        # Configure the Gemini API
        genai.configure(api_key=api_key)
        
        # Use Gemini 2.0 Flash to create a better prompt for image generation
        model = genai.GenerativeModel('gemini-2.0-flash')
        
        prompt_enhancement_text = f"""
        I need to generate an inspiring image based on these key words: "{key_words_text}"
        
        Create a detailed image generation prompt that will produce a beautiful, uplifting image.
        The image should be colorful, positive, and visually appealing.
        Include specific details about colors, composition, lighting, and mood.
        Make it suitable for sharing on social media as an inspirational post.
        The image must include ONLY the text "{key_words_text}", no other text.
        
        Respond with only the image generation prompt, no other text. Be as detailed as possible.
        """
        
        # Get enhanced prompt
        response = model.generate_content(prompt_enhancement_text)
        enhanced_prompt = response.text.strip()
        log(f"Created enhanced image prompt: {enhanced_prompt}")
        
        # Try to use Imagen 3 via direct API call
        try:
            log("Generating image with Imagen 3 via direct API call...")
            
            # Create a more detailed prompt with text rendering instructions
            enhanced_prompt = f"""
            Create an inspiring and uplifting image based on these key words: "{key_words_text}"
            
            Include ONLY the text "{key_words_text}" rendered beautifully in the image, with these specifications:
            - The text should be clearly visible and integrated into the design
            - Use an attractive, modern font that matches the mood
            - The text color should contrast well with the background
            - The text size should be proportional to the image
            
            The overall image should be:
            - Colorful and vibrant with positive energy
            - High quality with good lighting and composition
            - Suitable for sharing on social media
            - Include elements that visually represent the meaning of the words
            - Do not include any watermarks or additional text
            """
            
            log(f"Created enhanced prompt with text rendering instructions")
            
            # Create a temporary JSON request file
            request_data = {
                "instances": [
                    {
                        "prompt": enhanced_prompt,
                        "sampleCount": 1,
                        "negativePrompt": "blurry, distorted, low quality, ugly, poor composition, bad text, illegible text, additional text, watermark, signature, logo",
                        "seed": random.randint(1, 10000),  # Random seed for variety
                        "aspectRatio": "1:1",
                        "watermark": False  # Disable watermarking
                    }
                ],
                "parameters": {
                    "sampleCount": 1
                }
            }
            
            # Save request to a temporary file
            temp_request_file = tempfile.NamedTemporaryFile(delete=False, suffix='.json')
            with open(temp_request_file.name, 'w') as f:
                json.dump(request_data, f)
            
            # Build the curl command
            curl_command = f"""
            curl -X POST \
                 -H "Authorization: Bearer $(gcloud auth print-access-token)" \
                 -H "Content-Type: application/json; charset=utf-8" \
                 -d @{temp_request_file.name} \
                 "https://us-central1-aiplatform.googleapis.com/v1/projects/trusty-wavelet-447314-d1/locations/us-central1/publishers/google/models/imagen-3.0-generate-002@default:predict"
            """
            
            # Execute the curl command
            log("Executing API request to Imagen 3...")
            result = subprocess.run(curl_command, shell=True, capture_output=True, text=True)
            
            if result.returncode != 0:
                log(f"Error executing curl command: {result.stderr}")
                raise Exception(f"Imagen API call failed: {result.stderr}")
            
            # Parse the JSON response
            response_data = json.loads(result.stdout)
            
            # Extract the base64-encoded image
            if 'predictions' in response_data and len(response_data['predictions']) > 0:
                base64_image = response_data['predictions'][0]['bytesBase64Encoded']
                
                # Decode the base64 image
                image_data = base64.b64decode(base64_image)
                
                # Save to a temporary file
                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
                with open(temp_file.name, 'wb') as f:
                    f.write(image_data)
                
                # Validate the image
                try:
                    from PIL import Image
                    img = Image.open(temp_file.name)
                    img.verify()  # Verify it's a valid image
                    log("Successfully validated image file")
                    
                    log("Successfully generated image with Imagen 3")
                    st.success("✅ Successfully generated image with Imagen 3")
                    
                    # Upload to Google Cloud Storage
                    image_url = upload_to_gcs(temp_file.name, log_callback)
                    
                    # Store both the local path and the URL
                    return temp_file.name, image_url
                except Exception as img_error:
                    log(f"Invalid image file generated: {str(img_error)}")
                    os.remove(temp_file.name)  # Clean up the invalid file
                    raise Exception(f"Generated file is not a valid image: {str(img_error)}")
            else:
                log("No image data found in the API response")
                raise Exception("No image data in response")
                
        except Exception as api_error:
            log(f"Imagen 3 API call failed: {str(api_error)}")
            
            # Try alternative approach with jq for parsing (similar to your working curl command)
            try:
                log("Trying alternative approach with jq...")
                
                # Create output image path
                output_image_path = tempfile.NamedTemporaryFile(delete=False, suffix='.png').name
                
                # Use the same enhanced prompt with text rendering that we created earlier
                log(f"Using enhanced prompt with key words: '{key_words_text}'")
                
                # Build the full command with jq and base64 decoding
                full_command = f"""
                curl -X POST \
                     -H "Authorization: Bearer $(gcloud auth print-access-token)" \
                     -H "Content-Type: application/json; charset=utf-8" \
                     -d @{temp_request_file.name} \
                     "https://us-central1-aiplatform.googleapis.com/v1/projects/trusty-wavelet-447314-d1/locations/us-central1/publishers/google/models/imagen-3.0-generate-002@default:predict" \
                     | jq -r '.predictions[0].bytesBase64Encoded' | base64 --decode > {output_image_path}
                """
                
                # Execute the command
                log("Executing alternative command...")
                subprocess.run(full_command, shell=True, check=True)
                
                # Check if the image was created successfully and is a valid image
                if os.path.exists(output_image_path) and os.path.getsize(output_image_path) > 0:
                    # Validate the image file
                    try:
                        from PIL import Image
                        # Try to open the image to verify it's valid
                        img = Image.open(output_image_path)
                        img.verify()  # Verify it's a valid image
                        log("Successfully validated image file")
                        
                        # Upload to Google Cloud Storage
                        image_url = upload_to_gcs(output_image_path, log_callback)
                        
                        log("Successfully generated image with Imagen 3 (alternative method)")
                        st.success("✅ Successfully generated image with Imagen 3")
                        return output_image_path, image_url
                    except Exception as img_error:
                        log(f"Invalid image file generated: {str(img_error)}")
                        os.remove(output_image_path)  # Clean up the invalid file
                        raise Exception(f"Generated file is not a valid image: {str(img_error)}")
                else:
                    raise Exception("Image file was not created properly")
                    
            except Exception as alt_error:
                log(f"Alternative approach failed: {str(alt_error)}")
                log("Falling back to default image...")
                return get_default_image()
            
    except Exception as e:
        log(f"Image generation failed: {str(e)}. Using default image.")
        return get_default_image()

# Streamlit UI
def create_streamlit_ui():
    # Set page config
    st.set_page_config(page_title="Instagram Affirmation Sender", page_icon="✨", layout="wide")
    
    # Function to set background image
    def set_background_image(image_file='output_image.jpeg'):
        try:
            import base64
            from pathlib import Path
            
            # Check if the image file exists, if not use a default image
            if not Path(image_file).exists():
                # Use a default ocean image from Unsplash
                import requests
                from io import BytesIO
                
                response = requests.get("https://images.unsplash.com/photo-1507525428034-b723cf961d3e")
                if response.status_code == 200:
                    image_data = BytesIO(response.content).getvalue()
                else:
                    # If the request fails, return without setting a background
                    return
            else:
                # Read the local image file
                with open(image_file, "rb") as f:
                    image_data = f.read()
            
            # Encode the image to base64
            b64_encoded = base64.b64encode(image_data).decode()
            
            # Set the background image with CSS and add opacity control
            st.markdown(
                f"""
                <style>
                .stApp {{
                    background-image: url(data:image/png;base64,{b64_encoded});
                    background-size: cover;
                    background-position: center;
                    background-repeat: no-repeat;
                    position: relative;
                }}
                
                /* Add a pseudo-element with dark background for opacity control */
                .stApp::before {{
                    content: "";
                    position: absolute;
                    top: 0;
                    left: 0;
                    width: 100%;
                    height: 100%;
                    background-color: rgba(0, 0, 0, 0.5); /* Dark overlay with 50% opacity */
                    z-index: 0;
                    pointer-events: none; /* Allow clicks to pass through */
                }}
                
                /* Ensure all Streamlit elements are above the overlay */
                .stApp > * {{
                    position: relative;
                    z-index: 1;
                }}
                </style>
                """,
                unsafe_allow_html=True
            )
        except Exception as e:
            print(f"Error setting background image: {str(e)}")
    
    # Set the background image
    set_background_image()
    
    # Add custom CSS for styling
    st.markdown("""
    <style>
    /* Make all text white by default */
    .stApp {
        color: white !important;
    }
    
    /* Make all paragraphs, labels, and other text elements white */
    p, label, span, div, h1, h2, h3, h4, h5, h6, .stMarkdown, .stText, .stTextInput > label, 
    .stSelectbox > label, .stRadio > label, .stExpander > label {
        color: white !important;
    }
    
    /* Make input text white */
    .stTextInput > div > div > input, .stSelectbox > div > div > div {
        color: white !important;
    }
    
    /* Dark sidebar styling */
    [data-testid="stSidebar"] {
        background-color: #1E1E1E !important;
        border-right: 1px solid rgba(255, 255, 255, 0.1) !important;
    }
    
    /* Input fields in sidebar - black text on light background for better visibility */
    [data-testid="stSidebar"] .stTextInput > div > div > input,
    [data-testid="stSidebar"] .stTextArea > div > div > textarea {
        background-color: rgba(255, 255, 255, 0.8) !important;
        color: black !important;
        border-color: rgba(255, 255, 255, 0.2) !important;
    }
    
    /* Dropdown fields in sidebar */
    [data-testid="stSidebar"] .stSelectbox > div > div > div {
        background-color: rgba(255, 255, 255, 0.8) !important;
        color: black !important;
    }
    
    /* Dropdown options in sidebar */
    [data-testid="stSidebar"] .stSelectbox > div > div > div > div {
        background-color: rgba(255, 255, 255, 0.8) !important;
        color: black !important;
    }
    
    /* Ensure dropdown options are black */
    [data-testid="stSidebar"] .stSelectbox ul {
        background-color: white !important;
    }
    
    [data-testid="stSidebar"] .stSelectbox ul li {
        color: black !important;
    }
    
    /* Ensure the selected option in dropdown is black */
    [data-testid="stSidebar"] .stSelectbox > div > div > div > div[data-baseweb="select"] > div,
    [data-testid="stSidebar"] .stSelectbox > div > div > div > div[data-baseweb="select"] span {
        color: black !important;
    }
    
    /* Radio buttons in sidebar - keep text white but make selected state more visible */
    [data-testid="stSidebar"] .stRadio > div {
        color: white !important;
    }
    
    [data-testid="stSidebar"] .stRadio > div > div > label > div {
        color: white !important;
    }
    
    /* Sidebar header */
    [data-testid="stSidebarNav"] {
        background-color: #1E1E1E !important;
    }
    
    /* Style for the dark overlay containers */
    .overlay {
        background-color: rgba(0, 0, 0, 0.8); /* Increased opacity from 0.7 to 0.8 */
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2); /* Added shadow for better contrast */
    }
    
    .title {
        color: white;
        text-align: center;
        font-weight: bold;
    }
    
    .subtitle {
        color: #f0f0f0;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    /* Style for buttons */
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
        border: none;
        padding: 0.5rem 1rem;
        border-radius: 5px;
    }
    
    .stButton>button:hover {
        background-color: #45a049;
    }
    
    /* Style for expander */
    .stExpander {
        border: 1px solid rgba(255, 255, 255, 0.2);
        border-radius: 5px;
    }
    
    /* Style for text in the sidebar */
    .css-1d391kg, .css-1lcbmhc, .css-1wrcr25 {
        color: white !important;
    }
    
    /* Make sure all links are visible */
    a {
        color: #4CAF50 !important;
    }
    
    /* Style for input fields */
    input, select, textarea {
        background-color: rgba(255, 255, 255, 0.1) !important;
        border: 1px solid rgba(255, 255, 255, 0.2) !important;
        color: white !important;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Create a container with dark overlay for the main content
    with st.container():
        st.markdown('<div class="overlay">', unsafe_allow_html=True)
        st.markdown('<h1 class="title">✨ Instagram Affirmation Messenger</h1>', unsafe_allow_html=True)
        st.markdown('<p class="subtitle">Send AI-generated affirmations and images to your Instagram friends</p>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Initialize session state variables if they don't exist
    if 'affirmation_generated' not in st.session_state:
        st.session_state.affirmation_generated = False
    if 'logs' not in st.session_state:
        st.session_state.logs = []
    # Add session state for form inputs
    if 'gemini_api_key' not in st.session_state:
        st.session_state.gemini_api_key = ""
    if 'instagram_username' not in st.session_state:
        st.session_state.instagram_username = ""
    if 'instagram_password' not in st.session_state:
        st.session_state.instagram_password = ""
    if 'recipients' not in st.session_state:
        st.session_state.recipients = ""
    if 'theme' not in st.session_state:
        st.session_state.theme = "positivity"
    if 'method' not in st.session_state:
        st.session_state.method = "Selenium (Recommended)"

    # Create callback functions to update session state
    def update_api_key():
        st.session_state.gemini_api_key = st.session_state.api_key_input

    def update_username():
        st.session_state.instagram_username = st.session_state.username_input

    def update_password():
        st.session_state.instagram_password = st.session_state.password_input

    def update_recipients():
        st.session_state.recipients = st.session_state.recipients_input

    def update_theme():
        st.session_state.theme = st.session_state.theme_input

    def update_method():
        st.session_state.method = st.session_state.method_input

    # Sidebar for configuration
    with st.sidebar:
        st.markdown('<div class="overlay">', unsafe_allow_html=True)
        st.header("Configuration")
        
        # Use session state to persist values
        gemini_api_key = st.text_input(
            "Gemini API Key", 
            type="password",
            key="api_key_input",
            value=st.session_state.gemini_api_key,
            on_change=update_api_key
        )
        
        instagram_username = st.text_input(
            "Instagram Username",
            key="username_input",
            value=st.session_state.instagram_username,
            on_change=update_username
        )
        
        instagram_password = st.text_input(
            "Instagram Password", 
            type="password",
            key="password_input",
            value=st.session_state.instagram_password,
            on_change=update_password
        )
        
        recipients = st.text_input(
            "Recipients (comma-separated usernames)",
            key="recipients_input",
            value=st.session_state.recipients,
            on_change=update_recipients
        )
        
        theme = st.selectbox(
            "Affirmation Theme",
            ["positivity", "motivation", "success", "happiness", "gratitude", "self-love", "mindfulness"],
            key="theme_input",
            index=["positivity", "motivation", "success", "happiness", "gratitude", "self-love", "mindfulness"].index(st.session_state.theme),
            on_change=update_theme
        )
        
        method = st.radio(
            "Sending Method",
            ["Selenium (Recommended)", "Instabot"],
            key="method_input",
            index=0 if st.session_state.method == "Selenium (Recommended)" else 1,
            on_change=update_method
        )
        
        st.markdown("---")
        st.markdown("### Instructions")
        st.markdown("""
        1. Enter your Gemini API key (get one from [Google AI Studio](https://ai.google.dev/))
        2. Enter your Instagram credentials and recipient usernames
        3. Choose an affirmation theme
        4. Click 'Generate Affirmation & Image' to preview
        5. Review the generated content
        6. Click 'Send to Instagram' when ready
        """)
        
        st.markdown("---")
        st.markdown("### About API Quota Errors")
        st.markdown("""
        If you see quota errors, the app will use default affirmations and images.
        To avoid quota issues:
        - Create a new API key
        - Wait a few hours before trying again
        - Use the free tier responsibly
        """)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Main content area with dark overlay
    with st.container():
        st.markdown('<div class="overlay">', unsafe_allow_html=True)
        # Step 1: Generate content
        st.header("Step 1: Generate Affirmation & Image")
        
        if st.button("Generate New Affirmation & Image"):
            if not gemini_api_key:
                st.error("Please enter your Gemini API key in the sidebar")
            else:
                with st.spinner("Generating affirmation and image..."):
                    try:
                        # Define a log callback function to update the UI
                        def log_callback(message):
                            st.session_state.logs.append(message)
                        
                        # Step 1: Generate affirmation with Gemini
                        affirmation = generate_affirmation(gemini_api_key, theme)
                        st.session_state.affirmation = affirmation
                        
                        # Step 2: Generate image with Imagen 3
                        result = generate_image_with_imagen(gemini_api_key, affirmation, log_callback)
                        
                        if isinstance(result, tuple) and len(result) == 2:
                            # We have both local path and URL
                            image_path, image_url = result
                            st.session_state.image_path = image_path
                            st.session_state.image_url = image_url
                        else:
                            # Just local path
                            st.session_state.image_path = result
                            st.session_state.image_url = None
                        
                        # Mark as generated
                        st.session_state.affirmation_generated = True
                        
                        st.success("Generation complete! Review your content below.")
                    
                    except Exception as e:
                        st.error(f"Error generating content: {str(e)}")
                        
                        # Provide fallback content
                        default_affirmations = {
                            "positivity": "You are filled with unlimited potential. Every day brings new opportunities for growth.",
                            "motivation": "You have the strength to overcome any challenge. Keep pushing forward toward your dreams.",
                            "success": "Your hard work is creating the success you deserve. Celebrate each step of your journey.",
                            "happiness": "Joy lives within you, not in external circumstances. You deserve to be happy right now.",
                            "gratitude": "Your life is filled with blessings worth celebrating. Take a moment to appreciate all that you have.",
                            "self-love": "You are worthy of love and respect exactly as you are. Embrace your unique qualities.",
                            "mindfulness": "This present moment is a gift. Breathe deeply and connect with the peace that's always within you."
                        }
                        
                        st.session_state.affirmation = default_affirmations.get(theme, "You are capable of amazing things. Today is full of possibilities.")
                        st.session_state.image_path = get_default_image()
                        st.session_state.image_url = None
                        st.session_state.affirmation_generated = True
                        
                        st.warning("Using default content due to API error. You can still proceed.")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Display the generated content (if available)
    if st.session_state.affirmation_generated:
        with st.container():
            st.markdown('<div class="overlay">', unsafe_allow_html=True)
            st.header("Preview")
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.subheader("Affirmation:")
                st.write(st.session_state.affirmation)
            
            with col2:
                if hasattr(st.session_state, 'image_path') and st.session_state.image_path and os.path.exists(st.session_state.image_path):
                    st.subheader("Generated Image:")
                    try:
                        # Try to open the image to verify it's valid before displaying
                        from PIL import Image
                        img = Image.open(st.session_state.image_path)
                        # If it opens successfully, display it
                        st.image(st.session_state.image_path, use_column_width=True)
                    except Exception as e:
                        st.warning(f"Cannot display image: {str(e)}. A new image will be generated when sending.")
                        # Remove the invalid image path from session state
                        st.session_state.image_path = None
                else:
                    st.warning("Image generation failed. Only the affirmation text will be sent.")
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Step 2: Send to Instagram
        with st.container():
            st.markdown('<div class="overlay">', unsafe_allow_html=True)
            st.header("Step 2: Send to Instagram")
            
            # Create columns for the send button and status
            send_col, status_col = st.columns([1, 2])
            
            with send_col:
                if st.button("Send to Instagram"):
                    if not instagram_username or not instagram_password or not recipients:
                        st.error("Please fill in all Instagram credentials and recipient information")
                    else:
                        # Create a progress bar
                        progress_bar = st.progress(0)
                        status_message = st.empty()
                        
                        # Function to update logs in the UI
                        def update_log(message):
                            st.session_state.logs.append(message)
                            status_message.text(message)
                        
                        with st.spinner("Sending messages to Instagram..."):
                            affirmation = st.session_state.affirmation
                            image_path = st.session_state.image_path if hasattr(st.session_state, 'image_path') else None
                            image_url = st.session_state.image_url if hasattr(st.session_state, 'image_url') else None
                            
                            # Update progress
                            progress_bar.progress(10)
                            update_log("Starting Instagram automation...")
                            
                            if method == "Selenium (Recommended)":
                                success, message = automate_instagram_messages_selenium(
                                    instagram_username, instagram_password, recipients, 
                                    affirmation, image_path, update_log, image_url
                                )
                            else:
                                success, message = automate_instagram_messages_instabot(
                                    instagram_username, instagram_password, recipients, 
                                    affirmation, image_path, update_log, image_url
                                )
                            
                            # Update progress to complete
                            progress_bar.progress(100)
                            
                            if success:
                                st.success(message)
                            else:
                                st.error(message)
            
            with status_col:
                st.subheader("Log")
                
                # Create a dropdown for logs instead of showing them directly
                with st.expander("Click to view logs", expanded=False):
                    log_container = st.container()
                    
                    # Display logs
                    with log_container:
                        if st.session_state.logs:
                            for log in st.session_state.logs:
                                st.text(log)
                        else:
                            st.text("No logs available yet.")
            st.markdown('</div>', unsafe_allow_html=True)

# Update the initialize_genai_client function
def initialize_genai_client(api_key):
    # Initialize the client with the API key
    client = genai.Client(api_key=api_key)
    
    # Try to list available models
    try:
        # The correct method is to use client.list_models()
        models = client.list_models()
        model_names = [model.name for model in models]
        st.info(f"Available models: {', '.join(model_names)}")
    except Exception as e:
        st.warning(f"Could not list available models: {str(e)}")
    
    return client

# Update the upload_to_gcs function to handle uniform bucket-level access
def upload_to_gcs(image_path, log_callback=None):
    def log(message):
        print(message)
        if log_callback:
            log_callback(message)
    
    try:
        from google.cloud import storage
        import os
        import json
        import base64
        import tempfile
        import datetime
        
        # Extract the service account key from request.json
        log("Extracting service account key...")
        try:
            with open("./request.json", "r") as f:
                request_data = json.load(f)
            
            # The privateKeyData field contains the base64-encoded service account key
            if "privateKeyData" in request_data:
                # Decode the base64-encoded key
                key_json = base64.b64decode(request_data["privateKeyData"]).decode('utf-8')
                
                # Save to a temporary file
                key_file = tempfile.NamedTemporaryFile(delete=False, suffix='.json')
                with open(key_file.name, 'w') as f:
                    f.write(key_json)
                
                # Set the credentials file path
                os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = key_file.name
                log(f"Service account key extracted and saved to {key_file.name}")
            else:
                log("privateKeyData field not found in request.json")
                return None
        except Exception as e:
            log(f"Error extracting service account key: {str(e)}")
            return None
        
        log("Uploading image to Google Cloud Storage...")
        
        # Set up the client
        storage_client = storage.Client()
        
        # Get or create the bucket
        bucket_name = "instagram-affirmation-images"
        try:
            bucket = storage_client.get_bucket(bucket_name)
            log(f"Using existing bucket: {bucket_name}")
        except Exception:
            log(f"Bucket {bucket_name} not found. Creating new bucket...")
            bucket = storage_client.create_bucket(bucket_name, location="us-central1")
            log(f"Created new bucket: {bucket_name}")
            
            # Make the bucket publicly readable
            try:
                policy = bucket.get_iam_policy(requested_policy_version=3)
                policy.bindings.append({
                    "role": "roles/storage.objectViewer",
                    "members": ["allUsers"]
                })
                bucket.set_iam_policy(policy)
                log("Made bucket publicly readable")
            except Exception as e:
                log(f"Error making bucket public: {str(e)}")
        
        # Generate a unique blob name
        import uuid
        blob_name = f"affirmation_image_{uuid.uuid4()}.png"
        
        # Upload the file
        blob = bucket.blob(blob_name)
        blob.upload_from_filename(image_path)
        log("Image uploaded successfully")
        
        # Generate a signed URL that expires after 7 days
        try:
            # Generate a signed URL with a 7-day expiration
            signed_url = blob.generate_signed_url(
                version="v4",
                expiration=datetime.timedelta(days=7),
                method="GET"
            )
            log(f"Generated signed URL with 7-day expiration")
            
            # Shorten the URL for easier sharing
            short_url = shorten_url(signed_url, log_callback)
            log(f"Final image URL: {short_url}")
            return short_url
        except Exception as e:
            log(f"Error generating signed URL: {str(e)}")
            
            # Fallback to direct URL (which may not work if bucket is not public)
            direct_url = f"https://storage.googleapis.com/{bucket_name}/{blob_name}"
            log(f"Falling back to direct URL")
            
            # Try to shorten the direct URL too
            short_direct_url = shorten_url(direct_url, log_callback)
            log(f"Final direct URL: {short_direct_url}")
            return short_direct_url
    
    except Exception as e:
        log(f"Error uploading to Google Cloud Storage: {str(e)}")
        return None

# Add a URL shortening function
def shorten_url(long_url, log_callback=None):
    def log(message):
        print(message)
        if log_callback:
            log_callback(message)
    
    try:
        import requests
        import json
        
        log("Shortening URL...")
        
        # Try TinyURL API first (no API key required)
        try:
            tinyurl_api = f"https://tinyurl.com/api-create.php?url={long_url}"
            response = requests.get(tinyurl_api, timeout=10)
            
            if response.status_code == 200 and response.text.startswith("https://"):
                short_url = response.text
                log(f"URL shortened with TinyURL: {short_url}")
                return short_url
        except Exception as tinyurl_error:
            log(f"TinyURL shortening failed: {str(tinyurl_error)}")
        
        # Try Bitly as fallback (requires API key)
        try:
            # Check if we have a Bitly token in request.json
            try:
                with open("./request.json", "r") as f:
                    request_data = json.load(f)
                
                if "bitlyToken" in request_data:
                    bitly_token = request_data["bitlyToken"]
                    
                    # Use Bitly API
                    bitly_api = "https://api-ssl.bitly.com/v4/shorten"
                    headers = {
                        "Authorization": f"Bearer {bitly_token}",
                        "Content-Type": "application/json"
                    }
                    payload = {
                        "long_url": long_url,
                        "domain": "bit.ly"
                    }
                    
                    response = requests.post(bitly_api, headers=headers, json=payload, timeout=10)
                    
                    if response.status_code == 200:
                        result = response.json()
                        if "link" in result:
                            short_url = result["link"]
                            log(f"URL shortened with Bitly: {short_url}")
                            return short_url
            except Exception as bitly_error:
                log(f"Bitly shortening failed: {str(bitly_error)}")
        except:
            pass
        
        # If all shortening methods fail, return the original URL
        log("URL shortening failed, using original URL")
        return long_url
        
    except Exception as e:
        log(f"Error shortening URL: {str(e)}")
        return long_url

# Add these helper functions before the main function
def find_and_click_user_in_conversations(driver, username, log_callback=None, display_name=None):
    """Helper function to find and click on a user in the conversation list"""
    from selenium.webdriver.common.by import By
    
    def log(message):
        print(message)
        if log_callback:
            log_callback(message)
    
    try:
        # Wait for the conversation list to load
        time.sleep(3)
        
        # Extract display name from recipients if provided in format "username (display name)"
        if not display_name and "(" in username and ")" in username:
            parts = username.split("(")
            username = parts[0].strip()
            display_name = parts[1].replace(")", "").strip()
            log(f"Extracted username: {username} and display name: {display_name}")
        
        # Try multiple selectors to find the user by username
        user_selectors = [
            f"//span[contains(text(), '{username}')]",
            f"//div[contains(text(), '{username}')]",
            f"//a[contains(@href, '{username}')]",
            f"//div[contains(@aria-label, '{username}')]",
            f"//div[contains(@role, 'button')][contains(., '{username}')]",
            f"//div[contains(@class, 'conversation')][contains(., '{username}')]"
        ]
        
        # If we have a display name, add selectors for it too
        if display_name:
            display_name_selectors = [
                f"//span[contains(text(), '{display_name}')]",
                f"//div[contains(text(), '{display_name}')]",
                f"//div[contains(@aria-label, '{display_name}')]",
                f"//div[contains(@role, 'button')][contains(., '{display_name}')]",
                f"//div[contains(@class, 'conversation')][contains(., '{display_name}')]"
            ]
            user_selectors.extend(display_name_selectors)
            log(f"Added selectors for display name: {display_name}")
        
        # Try each selector
        for selector in user_selectors:
            try:
                user_elements = driver.find_elements(By.XPATH, selector)
                if user_elements and len(user_elements) > 0:
                    log(f"Found user with selector: {selector}")
                    # Click on the first matching element
                    user_elements[0].click()
                    log(f"Clicked on user in conversation list")
                    time.sleep(3)
                    return True
            except Exception as e:
                log(f"Error with selector {selector}: {str(e)}")
                continue
        
        # If we couldn't find the user with specific selectors, try a more general approach
        # Look for any element that might contain the username or display name
        log("Trying general approach to find user in conversation list...")
        all_elements = driver.find_elements(By.XPATH, "//div")
        search_terms = [username.lower()]
        if display_name:
            search_terms.append(display_name.lower())
        
        for element in all_elements:
            try:
                element_text = element.text.lower()
                for term in search_terms:
                    if term in element_text:
                        log(f"Found element containing '{term}': {element.text}")
                        element.click()
                        log(f"Clicked on element containing '{term}'")
                        time.sleep(3)
                        return True
            except:
                continue
        
        # Take a screenshot for debugging
        screenshot_path = f"conversation_list_{username}.png"
        driver.save_screenshot(screenshot_path)
        log(f"Could not find user in conversation list. Saved screenshot to {screenshot_path}")
        
        return False
    except Exception as e:
        log(f"Error finding user in conversations: {str(e)}")
        return False

def ensure_conversation_selected(driver, username, log_callback=None):
    """Helper function to ensure the correct conversation is selected"""
    from selenium.webdriver.common.by import By
    
    def log(message):
        print(message)
        if log_callback:
            log_callback(message)
    
    try:
        # Extract display name if provided in format "username (display name)"
        display_name = None
        if "(" in username and ")" in username:
            parts = username.split("(")
            username = parts[0].strip()
            display_name = parts[1].replace(")", "").strip()
            log(f"Extracted username: {username} and display name: {display_name}")
        
        # Check if we're already in the correct conversation
        conversation_indicators = [
            f"//span[contains(text(), '{username}')]",
            f"//div[contains(text(), '{username}')]",
            f"//h1[contains(text(), '{username}')]"
        ]
        
        # Add display name indicators if available
        if display_name:
            display_name_indicators = [
                f"//span[contains(text(), '{display_name}')]",
                f"//div[contains(text(), '{display_name}')]",
                f"//h1[contains(text(), '{display_name}')]"
            ]
            conversation_indicators.extend(display_name_indicators)
        
        for indicator in conversation_indicators:
            try:
                elements = driver.find_elements(By.XPATH, indicator)
                if elements and len(elements) > 0:
                    log(f"Already in conversation with {username}")
                    return True
            except:
                continue
        
        # If we're not in the correct conversation, try to find and click on the user
        log(f"Not in conversation with {username}, trying to find and select")
        return find_and_click_user_in_conversations(driver, username, log_callback, display_name)
    except Exception as e:
        log(f"Error ensuring conversation is selected: {str(e)}")
        return False

# Add a function to handle the specific case for pixelwhisperss
def handle_special_accounts(username):
    """Map usernames to their display names for special cases"""
    username_map = {
        "pixelwhisperss": "pixelated dreams"
    }
    
    # Check if this username is in our special cases map
    if username.lower() in username_map:
        return username_map[username.lower()]
    return None

# Add a new function to find the message input field using multiple approaches
def find_message_input(driver, log_callback=None):
    """Use multiple approaches to find the Instagram message input field"""
    from selenium.webdriver.common.by import By
    from selenium.webdriver.common.keys import Keys
    from selenium.webdriver.common.action_chains import ActionChains
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    import time
    
    def log(message):
        print(message)
        if log_callback:
            log_callback(message)
    
    log("Attempting to find message input field with enhanced methods...")
    
    # Take a screenshot for debugging
    screenshot_path = "message_input_debug.png"
    driver.save_screenshot(screenshot_path)
    log(f"Saved debug screenshot to {screenshot_path}")
    
    # Approach 1: Try standard selectors
    input_selectors = [
        "//textarea[@placeholder='Message...']",
        "//div[@role='textbox']", 
        "//div[@contenteditable='true']",
        "//div[contains(@aria-label, 'Message')]",
        "//textarea[contains(@placeholder, 'Message')]",
        "//textarea",
        "//div[@data-testid='message-composer']",
        "//div[contains(@class, 'focus-visible')]",
        "//div[contains(@class, 'message-composer')]",
        "//div[contains(@class, 'text-input')]",
        "//div[contains(@class, 'composer')]",
        "//div[contains(@class, 'input')]",
        "//div[contains(@class, 'editor')]",
        "//div[contains(@class, 'editable')]",
        "//div[@role='combobox']",
        "//div[@role='input']",
        "//div[@data-lexical-editor='true']",
        "//div[@spellcheck='true']"
    ]
    
    for selector in input_selectors:
        try:
            log(f"Trying selector: {selector}")
            elements = driver.find_elements(By.XPATH, selector)
            if elements and len(elements) > 0:
                for element in elements:
                    try:
                        if element.is_displayed():
                            log(f"Found visible input with selector: {selector}")
                            element.click()
                            time.sleep(1)
                            return element
                    except:
                        continue
        except Exception as e:
            log(f"Error with selector {selector}: {str(e)}")
    
    # Approach 2: Try to find the message area first, then look for input within it
    log("Trying to find message area container first...")
    container_selectors = [
        "//div[contains(@aria-label, 'Conversation')]",
        "//div[@role='main']",
        "//section[@role='main']",
        "//div[contains(@class, 'message-container')]",
        "//div[contains(@class, 'conversation')]",
        "//div[contains(@class, 'chat')]",
        "//footer",
        "//div[@role='contentinfo']"
    ]
    
    for selector in container_selectors:
        try:
            containers = driver.find_elements(By.XPATH, selector)
            if containers and len(containers) > 0:
                for container in containers:
                    try:
                        if container.is_displayed():
                            log(f"Found message container with selector: {selector}")
                            
                            # Look for input elements within this container
                            for input_selector in input_selectors:
                                try:
                                    relative_selector = f".{input_selector}"
                                    input_elements = container.find_elements(By.XPATH, relative_selector)
                                    if input_elements and len(input_elements) > 0:
                                        for input_element in input_elements:
                                            if input_element.is_displayed():
                                                log(f"Found input within container using selector: {input_selector}")
                                                input_element.click()
                                                time.sleep(1)
                                                return input_element
                                except:
                                    continue
                            
                            # If we found a container but no input within it, try clicking the container itself
                            log("No input found in container, trying to click container")
                            container.click()
                            time.sleep(1)
                            
                            # Try to find any clickable elements at the bottom of the container
                            try:
                                bottom_elements = container.find_elements(By.XPATH, ".//div")
                                # Sort elements by their vertical position (bottom first)
                                bottom_elements.sort(key=lambda e: e.location['y'], reverse=True)
                                
                                for element in bottom_elements[:5]:  # Try the 5 bottom-most elements
                                    try:
                                        if element.is_displayed():
                                            log("Clicking on element at bottom of container")
                                            element.click()
                                            time.sleep(1)
                                            active_element = driver.switch_to.active_element
                                            if active_element:
                                                return active_element
                                    except:
                                        continue
                            except:
                                pass
                    except:
                        continue
        except Exception as e:
            log(f"Error with container selector {selector}: {str(e)}")
    
    # Approach 3: Try clicking at the bottom of the screen
    log("Trying to click at the bottom of the screen...")
    try:
        # Get window size
        window_size = driver.get_window_size()
        width = window_size['width']
        height = window_size['height']
        
        # Try different positions at the bottom of the screen
        positions = [
            (width // 2, int(height * 0.9)),  # Center bottom
            (width // 2, int(height * 0.85)),  # Slightly higher
            (width // 2, int(height * 0.95)),  # Lower
            (int(width * 0.25), int(height * 0.9)),  # Left side
            (int(width * 0.75), int(height * 0.9))   # Right side
        ]
        
        for x, y in positions:
            log(f"Clicking at position: x={x}, y={y}")
            actions = ActionChains(driver)
            actions.move_by_offset(x, y).click().perform()
            time.sleep(1)
            
            # Check if we activated an input
            active_element = driver.switch_to.active_element
            if active_element:
                tag_name = active_element.tag_name.lower()
                if tag_name in ['textarea', 'input'] or active_element.get_attribute('contenteditable') == 'true':
                    log(f"Found input element after clicking at position: x={x}, y={y}")
                    return active_element
    except Exception as e:
        log(f"Error clicking at screen positions: {str(e)}")
    
    # Approach 4: Try using Tab key navigation
    log("Trying Tab key navigation...")
    try:
        # First click somewhere in the page
        body = driver.find_element(By.TAG_NAME, "body")
        body.click()
        
        # Press Tab multiple times to try to reach the input field
        for i in range(15):  # Try up to 15 tabs
            actions = ActionChains(driver)
            actions.send_keys(Keys.TAB)
            actions.perform()
            time.sleep(0.5)
            
            # Check if we found an input-like element
            active = driver.switch_to.active_element
            tag_name = active.tag_name.lower()
            
            if tag_name in ['textarea', 'input'] or active.get_attribute('contenteditable') == 'true':
                log(f"Found potential input field with tag: {tag_name} after {i+1} tabs")
                return active
    except Exception as e:
        log(f"Error with Tab navigation: {str(e)}")
    
    # Approach 5: Try JavaScript injection
    log("Trying JavaScript injection...")
    try:
        js_result = driver.execute_script("""
            // Try to find message input elements
            var possibleInputs = [
                document.querySelector('textarea[placeholder*="Message"]'),
                document.querySelector('div[role="textbox"]'),
                document.querySelector('div[contenteditable="true"]'),
                document.querySelector('textarea'),
                document.querySelector('div[data-testid="message-composer"]'),
                document.querySelector('div.focus-visible'),
                document.querySelector('div.message-composer'),
                document.querySelector('div.text-input'),
                document.querySelector('div.composer'),
                document.querySelector('div.input'),
                document.querySelector('div.editor'),
                document.querySelector('div.editable'),
                document.querySelector('div[role="combobox"]'),
                document.querySelector('div[role="input"]'),
                document.querySelector('div[data-lexical-editor="true"]'),
                document.querySelector('div[aria-label*="Message"]'),
                document.querySelector('div[aria-label*="Type"]'),
                document.querySelector('div[spellcheck="true"]'),
                // Try to find by class name containing these terms
                document.querySelector('div[class*="composer"]'),
                document.querySelector('div[class*="input"]'),
                document.querySelector('div[class*="message"]'),
                document.querySelector('div[class*="text"]'),
                document.querySelector('div[class*="editor"]')
            ];
            
            // Function to check if element is visible
            function isVisible(elem) {
                if (!elem) return false;
                const style = window.getComputedStyle(elem);
                return style.display !== 'none' && 
                       style.visibility !== 'hidden' && 
                       elem.offsetWidth > 0 && 
                       elem.offsetHeight > 0;
            }
            
            // Return the first visible non-null element
            for (var i = 0; i < possibleInputs.length; i++) {
                if (possibleInputs[i] && isVisible(possibleInputs[i])) {
                    possibleInputs[i].focus();
                    return true;
                }
            }
            
            // If we can't find by querySelector, try a more aggressive approach
            // Look for any element that might be an input
            var allDivs = document.querySelectorAll('div');
            for (var i = 0; i < allDivs.length; i++) {
                var div = allDivs[i];
                // Check if this div has characteristics of an input field
                if (isVisible(div) && (
                    div.getAttribute('contenteditable') === 'true' || 
                    div.getAttribute('role') === 'textbox' ||
                    div.getAttribute('role') === 'combobox' ||
                    div.getAttribute('role') === 'input' ||
                    div.getAttribute('data-lexical-editor') === 'true' ||
                    div.getAttribute('spellcheck') === 'true' ||
                    (div.className && (
                        div.className.includes('input') || 
                        div.className.includes('composer') || 
                        div.className.includes('editor') || 
                        div.className.includes('message') ||
                        div.className.includes('text')
                    ))
                )) {
                    div.focus();
                    return true;
                }
            }
            
            return false;
        """)
        
        if js_result:
            log("Found and focused input field via JavaScript")
            return driver.switch_to.active_element
    except Exception as e:
        log(f"Error with JavaScript injection: {str(e)}")
    
    # If all approaches fail, return None
    log("All approaches failed to find message input field")
    return None

# Run the Streamlit app
if __name__ == "__main__":
    create_streamlit_ui()