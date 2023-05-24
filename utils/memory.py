from textwrap import dedent
import json

from .logger import red, yel
from .misc import tokenize

def curate_previous_prompts(memorized_prompts):
    "auto disable passed example if too many tokens"

    # make sure they all have a disabled flag
    for i, mess in enumerate(memorized_prompts):
        if "disabled" not in memorized_prompts[i]:
            memorized_prompts[i]["disabled"] = False
            yel(f"Edited entry {i} to add the 'disabled' flag")
        assert mess["disabled"] in [True, False], "invalid value of disabled flag of a message"
        assert mess["role"] in ["system", "user", "assistant"], "invalid value of role value of message"
        assert "CONTEXT" not in mess["content"], "CONTEXT string found in message!"
        assert "TRANSCRIPT" not in mess["content"], "TRANSCRIPT string found in message!"
        memorized_prompts[i]["content"] = dedent(memorized_prompts[i]["content"]).strip()

    while True:
        tkns = 0
        dis_tkns = 0
        for m in memorized_prompts:
            if not m["disabled"]:
                tkns += len(tokenize(m["content"]))
            else:
                dis_tkns += len(tokenize(m["content"]))
        red(f"Nondisabled token count : {tkns} (total: {tkns + dis_tkns})")
        if tkns > 3500:
            red("Disabling oldest examples")
            for i, m in enumerate(memorized_prompts):
                if m["role"] == "system":
                    continue
                if i % 2 != 1:
                    continue
                if not m["disabled"]:
                    memorized_prompts[i]["disabled"] = True
                    assert memorized_prompts[i]["role"] == "user", "unexpected role not user"
                    memorized_prompts[i + 1]["disabled"] = True
                    assert memorized_prompts[i + 1]["role"] == "assistant", "unexpected role not assitant"
                    break

        else:
            break
    return memorized_prompts





with open("user_data/g/memories.json", "r") as f:
    memorized_prompts = json.load(f)


# check previous prompts just in case
if not memorized_prompts or not isinstance(memorized_prompts, list):
    print(memorized_prompts)
    raise SystemExit("Invalid memorized_prompts")
with open("audio_prompts.json", "w") as f:
    json.dump(memorized_prompts, f, indent=4)
memorized_prompts = curate_previous_prompts(memorized_prompts)

