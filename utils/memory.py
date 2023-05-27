import time
from pathlib import Path
from textwrap import dedent
import json

from .logger import whi, red, yel
from .misc import tokenize, transcript_template

global prev_prompts

default_system_prompt = [
        {
            "role": "system",
            "content": "You are Alfred, my excellent assistant who always exceeds my expactation. Your task today is the to transform audio transcripts from french medical flashcards into anki flashcards.\n\n            Here are the rules:\n                * always end your answers by \"END\".\n                * If you create several flashcards for one transcript, separate them with \"#####\".\n                * You are allowed to use medical acronyms.\n                * the flashcards have to be as concise as possible.\n                * the flashcards have to be close to the format \"[category], [question] ?<br/>[answer]\". I'll show you examples of good formats.\n                * the transcript are from french medical textbooks, it is also your job to correct misrecognized words using the other words and the supplied context.\n                * don't reply anything other than the answer ot your task\n                * if you're absolutely sure that you must ask me something: begin your answer by 'Alfred:' and I'll come immediately.",
            "timestamp": int(time.time()),
            "priority": 10,
            }
        ]
expected_mess_keys = ["role", "content", "timestamp", "priority", "tkn_len", "answer"]


def check_prompts(prev_prompts):
    "checks validity of the previous prompts"
    whi("Checking prompt validity")
    for i, mess in enumerate(prev_prompts):

        assert mess["role"] in ["system", "user"], "invalid value of role value of message"
        if mess["role"] == "user":
            assert "answer" in mess, "no answer key in message"

        assert "CONTEXT" not in mess["content"], "CONTEXT string found in message!"
        assert "TRANSCRIPT" not in mess["content"], "TRANSCRIPT string found in message!"

        if "timestamp" not in mess:
            mess["timestamp"] = i  # if no timestamp, at least keep the order
        assert isinstance(mess["timestamp"], int), "timestamp is not int!"

        if "priority" not in mess:
            mess["priority"] = 0
            if "importance" in mess:  # backward compatibility
                mess["priority"] = mess["importance"]
        assert isinstance(mess["priority"], int), "priority is not int!"
        assert mess["priority"] <= 10, "priority above 10 !"
        assert mess["priority"] >= 0, "priority under 0 !"

        if "tkn_len" not in mess:
            mess["tkn_len"] = len(tokenize(dedent(mess["content"]).strip()))
            if "answer" in mess:  # system prompt has no answer
                mess["tkn_len"] += len(tokenize(dedent(mess["answer"]).strip()))
            else:
                assert mess["role"] == "system", "expected system message here"
        assert isinstance(mess["tkn_len"], int), "tkn_len is not int!"
        assert mess["tkn_len"] > 0, "tkn_len under 0 !"

        # keep only the expected keys
        keys = [k for k in mess.keys() if k in expected_mess_keys]
        for k in keys:
            if k in prev_prompts[i]:
                prev_prompts[i][k] = mess[k]

        # make sure it's stripped
        prev_prompts[i]["content"] = dedent(prev_prompts[i]["content"]).strip()
        if "answer" in prev_prompts[i]:
            prev_prompts[i]["answer"] = dedent(prev_prompts[i]["answer"]).strip()

    return prev_prompts


def prompt_filter(prev_prompts, max_token):
    """goes through the list of previous prompts of the profile, check
    correctness of the key/values, then returns only what's under the maximum
    number of tokens for model"""

    assert max_token >= 500, "max_token should be above 500"
    assert max_token <= 3500, "max_token should be under 3500"

    timesorted_pr = sorted(prev_prompts, key=lambda x: x["timestamp"], reverse=True)
    syspr = [pr for pr in prev_prompts if pr["role"] == "system"]
    assert len(syspr) == 1, "Number of system prompts != 1"

    # add by decreasing priority and timestamp
    prio_vals = sorted(set([x["priority"] for x in prev_prompts]), reverse=True)
    tkns = syspr[0]["tkn_len"]
    dis_tkns = 0
    output_pr = [syspr[0]]
    for prio in prio_vals:
        red(f"priority: {prio}")  # debug
        for pr in timesorted_pr:
            if pr in output_pr:
                continue
            if pr["priority"] == prio:
                if not tkns + pr["tkn_len"] > max_token:
                    tkns += pr["tkn_len"]
                    output_pr.append(pr)
                else:
                    dis_tkns += pr["tkn_len"]
    red(f"Nondisabled token count : {tkns} (total: {tkns + dis_tkns})")
    yel(f"Number of prompts: '{len(prev_prompts)}'")
    assert len(output_pr) > 1 or len(prev_prompts) == 1, "invalid prompt output"
    return output_pr


def recur_improv(choice_profile, txt_audio, txt_whisp_prompt, txt_chatgpt_cloz, txt_context, priority, output):
    # checks that there is no , in profile, which would be the default value
    if "," in choice_profile:
        spl = [s for s in choice_profile.strip().split(",") if s.strip()]
        if not len(spl) == 1:
            raise Exception(f"profile with invalid value: '{choice_profile}'")
        else:
            choice_profile = spl[0]

    whi("Recursively improving")
    if not txt_audio:
        return "No audio transcripts found.\n\n" + output
    if not txt_chatgpt_cloz:
        return "No chatgpt cloze found.\n\n" + output
    if "\n" in txt_chatgpt_cloz:
        whi("Replaced newlines in txt_chatgpt_cloz")
        txt_chatgpt_cloz = txt_chatgpt_cloz.replace("\n", "<br/>")

    prev_prompts = load_prev_prompts(choice_profile)
    try:
        to_add = [
                {
                    "role": "user",
                    "content": transcript_template.replace("CONTEXT", txt_context).replace("TRANSCRIPT", txt_audio),
                    "timestamp": int(time.time()),
                    "priority": priority,
                    "answer": txt_chatgpt_cloz.replace("\n", "<br/>"),
                    }
                ]
        if to_add[0] in prev_prompts:
            return f"Already present in previous outputs!\n\n{output}"
        prev_prompts.extend(to_add)

        prev_prompts = check_prompts(prev_prompts)

        with open(f"profiles/{choice_profile}/memories.json", "w") as f:
            json.dump(prev_prompts, f, indent=4)
    except Exception as err:
        return f"Error during recursive improvement: '{err}'\n\n{output}"
    return f"Recursively improved: {len(prev_prompts)} total examples" + "\n\n" + output


def load_prev_prompts(profile):
    assert Path("profiles/").exists(), "profile directory not found"
    if Path(f"profiles/{profile}/memories.json").exists():
        with open(f"profiles/{profile}/memories.json", "r") as f:
            prev_prompts = json.load(f)
        prev_prompts = check_prompts(prev_prompts)
    else:
        red("No memories in profile found, creating it")
        prev_prompts = check_prompts(default_system_prompt.copy())
        with open(f"profiles/{profile}/memories.json", "w") as f:
            json.dump(prev_prompts, f, indent=4)

    return prev_prompts
