# Inf502 Project

## Mistral AI API Key Setup

### Obtain a Mistral AI API Key
To use the Mistral AI API, you need to obtain an API key from their official website:

1. Go to [Mistral AI's website](https://mistral.ai/).
2. Sign up or log in to your account.
3. Navigate to the API section and generate a new API key.
4. Copy the API key for later use.

### Store the API Key Securely
It is recommended to store the API key in a .env file to keep it secure and avoid hardcoding it in your code.

#### Creating the .env file

1. In your project directory, create a new file named .env if it does not already exist.
2. Open the .env file in a text editor and add the following line:

   sh
   MISTRAL_API_KEY=your_api_key_here

   Replace your_api_key_here with the actual API key you obtained.

To run:

1.Place images to train VAE in data/images/Images

2.run VAE.py (1 and 2 optional if only running Linear)

3.run SD.py(change model according to need, interp_model=LnearInterpolation() if Linear model wanted)

4.run Loss.py to see loss

In scratch: Our first ideas, failed attempts, etc.