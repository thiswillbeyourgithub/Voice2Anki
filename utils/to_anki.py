import urllib.request
import json
from .logger import red


def add_to_anki(
        body,
        source,
        note_metadata,
        tags,
        deck_name="Default",
        ):
    """create a new cloze directly in anki"""
    assert isinstance(tags, list), "tags have to be a list"
    return _call_anki(
            action="addNote",
            note={
                "deckName": deck_name,
                "modelName": "Clozolkor",
                "fields": {
                    "body": body,
                    "header": "",
                    "hint": "",
                    "more": "",
                    "source": source,
                    "source_extra": "",
                    "teacher": "",
                    "Nearest_neighbors": "",
                    "GPToAnkiMetadata": note_metadata,
                    },
                "tags": tags,
                "options": {"allowDuplicate": True},
               },
            )


def _call_anki(action, **params):
    """ bridge between local python libraries and AnnA Companion addon
    (a fork from anki-connect) """
    def request_wrapper(action, **params):
        return {'action': action, 'params': params, 'version': 6}

    requestJson = json.dumps(request_wrapper(action, **params)
                             ).encode('utf-8')

    try:
        response = json.load(urllib.request.urlopen(
            urllib.request.Request(
                'http://localhost:8765',
                requestJson)))
    except (ConnectionRefusedError, urllib.error.URLError) as e:
        red(f"{str(e)}: is Anki open and 'AnkiConnect' "
            "enabled? Firewall issue?")
        raise Exception(f"{str(e)}: is Anki open and 'AnkiConnect' "
                        "addon' enabled? Firewall issue?")

    if len(response) != 2:
        red('response has an unexpected number of fields')
        raise Exception('response has an unexpected number of fields')
    if 'error' not in response:
        red('response is missing required error field')
        raise Exception('response is missing required error field')
    if 'result' not in response:
        red('response is missing required result field')
        raise Exception('response is missing required result field')
    if response['error'] is not None:
        red(response['error'])
        raise Exception(response['error'])
    return response['result']


if __name__ == "__main__":
    print(
            add_to_anki(
                body="{{c1::test}}",
                source="truc",
                note_metadata="",
                tags=["test"],
                deck_name="Default"
                )
            )
    breakpoint()
