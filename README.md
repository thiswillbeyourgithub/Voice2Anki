# Voice2Anki

## What is this?
* Voice2Anki is a tool that leverages the power of ChatGPT to correct the transcription of Whisper to create anki flashcards. It allows to create many very high quality flashcards.

### Features
* Works in any language
* Works on any topic: LLM can correct Whisper efficiently
* **Adapts to you** If a flashcard was not exactly like you wanted, correct it (manually or using your voice) then save it for the future and the LLM will use it for the next cards. You can save many examples like this as embeddings and keywords are used to find the most relevant example for your use case at each prompt.
* You can specify anki tags, marks, deck etc.
* If you supply one or several images, it will be OCR'ed then saved in the flashcard in a source field. A special algorithm is used to keep the formatting of the image in the OCR step.
* Supports ChatGPT, GPT-4, replicate models thanks to [litellm](https://docs.litellm.ai/), and many more support can be added.
* Supports multiple profile. Making it handy for various use.

## Getting started
* clone this repo
* Make sure you have python 3.11 (needed for asyncio.timeout)
* Install the dependencies: `python -m pip install -r requirements.txt`
* Anki must be open and with addon [AnkiConnect](https://ankiweb.net/shared/info/2055492159) enabled.
* `python Voice2Anki.py --browser`
    * `--open_browser` opens the browser on the interface.
    * `--authentication` enables the authentication panel. user/password have to be edited in `Voice2Anki.py`.
    * `--localnetwork` to make the interface accessible to your local network. Use `ifconfig` to figure out your local IP adress and connect to it using `https://[IP]:7860` (don't forget the http**s** ). You can use that to make it accessible from the global internet if you configure port forwarding from your router. Otherwise it's only accessible from the computer.
    * `--debug` to increase verbosity.
    * **CAREFUL** if you add `--share`, the interface will be forwarded to Hugging Face's website and accessible via a URL for 72 hours. Handy if you want to use Voice2Anki on mobile but can have privacy and security implications.
* open your browser at the URL shown in the output.
* The first thing to do is to enter a profile name in the `profile` field. This will automatically fill the other fields with the latest value. Then also go enter the API key.

## Notes
* If using SSL, the certificates will probably be self signed, you'll likely have to tick a few checkbox on your browser to access the site.
* It's apparently way less CPU intensive to use Chromium than to use firefox according to my limited testing with a heavily modified firefox.
* For now running the script creates temporary .wav files that are deleted at each startup automatically. This will be fixed eventually.
* tips: if you want to quickly have high quality card, add the end of the recording mention notes to alfred like "Note à Alfred: fait 3 cartes sur cette notion" or "Note à Alfred: fait une carte de liste". Then simply manually delete from the transcript that you guided Alfred and save the prompt as a good example of Alfred doing what you wanted.
* The memories prompts are stored in your profile folder.
* To update the app, you just have to do `git pull`
* Reach out if you have any issue.
* Feedbacks (of any nature) are much appreciated.


## Gallery
* Anki backend:
  * ![](./docs/anki_screenshot.png)
