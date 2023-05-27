# WhisperToAnki

## Getting started
    * Install the dependencies: `python -m pip install -r requirements.txt`
    * Make sure Anki is open and with addon [AnkiConnect](https://ankiweb.net/shared/info/2055492159) enabled.
    * Copy your OpenAI API key to a file called `API_KEY.txt`

## Usage
    * `python __init__.py --browser --noauth`
        * `--browser` opens the browser on the interface.
        * `--noauth` disables the authentication panel. Otherwise, the user/password can be edited in the __init__.py file.
        * `--localnetwork` to make the interface accessible to your local network. Use `ifconfig` to figure out your local IP adress and connect to it using `https://[IP]:7860` (don't forget the http**s** ). You can use that to make it accessible from the global internet if you configure port forwarding from your router. Otherwise it's only accessible from the computer.
        * `--debug` to increase verbosity.
        * **CAREFUL** if you add `--share`, the interface will be forwarded to Hugging Face's website and accessible via a URL for 72 hours. Handy if you want to use WhisperToAnki on mobile but can have privacy and security implications.

