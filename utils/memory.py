from pathlib import Path
import sys
from textwrap import dedent
import json

from .logger import whi, red, yel
from .misc import tokenize, transcript_template


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


def recur_improv(txt_audio, txt_whisp_prompt, txt_chatgpt_cloz, txt_context, output):
    whi("Recursively improving")
    global memorized_prompts
    if not txt_audio:
        return "No audio transcripts found.\n\n" + output
    if not txt_chatgpt_cloz:
        return "No chatgpt cloze found.\n\n" + output
    if "\n" in txt_chatgpt_cloz:
        whi("Replaced newlines in txt_chatgpt_cloz")
        txt_chatgpt_cloz = txt_chatgpt_cloz.replace("\n", "<br/>")

    try:
        assert len(memorized_prompts) % 2 == 1, "invalid length of new prompts before even updating it"
        to_add = [
                {
                    "role": "user",
                    "content": transcript_template.replace("CONTEXT", txt_context).replace("TRANSCRIPT", txt_audio),
                    "disabled": False,
                    },
                {
                    "role": "assistant",
                    "content": txt_chatgpt_cloz.replace("\n", "<br/>"),
                    "disabled": False,
                    }
                ]
        if to_add[0] in memorized_prompts:
            return f"Already present in previous outputs!\n\n{output}"
        if to_add[1] in memorized_prompts:
            return f"Already present in previous outputs!\n\n{output}"
        memorized_prompts.extend(to_add)

        memorized_prompts = curate_previous_prompts(memorized_prompts)

        assert len(memorized_prompts) % 2 == 1, "invalid length of new prompts"

        with open("audio_prompts.json", "w") as f:
            json.dump(memorized_prompts, f, indent=4)
    except Exception as err:
        return f"Error during recursive improvement: '{err}'\n\n{output}"
    return f"Recursively improved: {len(memorized_prompts)} total examples" + "\n\n" + output




args = sys.argv[1:]
collated = " ".join(args)
if "--username=" not in collated:
    raise Exception("No --username=USERNAME argument found")
if "--profile=" not in collated:
    raise Exception("No --profile=PROFILE argument found")
for ar in args:
    if "--username=" in ar:
        username = ar.replace("--username=", "")
    if "--profile=" in ar:
        profile = ar.replace("--profile=", "")

assert Path(f"user_data/{username}").exists(), "No user directory found"
assert Path(f"user_data/{username}/{profile}.json").exists(), "No user profile found"
with open(f"user_data/{username}/{profile}.json", "r") as f:
    memorized_prompts = json.load(f)


# check previous prompts just in case
if not memorized_prompts or not isinstance(memorized_prompts, list):
    print(memorized_prompts)
    raise SystemExit("Invalid memorized_prompts")
with open("audio_prompts.json", "w") as f:
    json.dump(memorized_prompts, f, indent=4)
memorized_prompts = curate_previous_prompts(memorized_prompts)

