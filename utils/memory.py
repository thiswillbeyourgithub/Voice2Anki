import random
import time
from pathlib import Path
from textwrap import dedent
import json

from .logger import whi, red, yel
from .misc import tokenize, transcript_template

global prev_prompts

default_system_prompt = {
            "role": "system",
            "content": dedent("""You are my excellent assistant Alfred. You always exceed my expectations. Your task today is the to transform audio transcripts into Anki flashcards.

            Here are the rules:
                * always end your replies by "END".
                * If you create several flashcards for one transcript, separate them with "#####".
                * You are encouraged to use common acronyms.
                * the transcript can be of poor quality, it is your job to correct transcription errors using the context.
                * if you're absolutely certain that you can't accomplish your task: begin your answer by 'Alfred:' and I'll take a look immediately."""),
            "timestamp": int(time.time()),
            "priority": -1,  # the only prompt that has priority of -1 is the system prompt
            }
expected_mess_keys = ["role", "content", "timestamp", "priority", "tkn_len", "answer", "llm_model", "tts_model"]


def check_prompts(prev_prompts):
    "checks validity of the previous prompts"
    whi("Checking prompt validity")
    for i, mess in enumerate(prev_prompts):

        if mess["role"] == "user":
            assert "answer" in mess, "no answer key in message"
        elif mess["role"] == "system":
            mess["content"] = default_system_prompt["content"]
            mess["priority"] = default_system_prompt["priority"]
        else:
            raise ValueError("role of previous prompt is not user or system")

        assert "CONTEXT" not in mess["content"], "CONTEXT string found in message!"
        assert "TRANSCRIPT" not in mess["content"], "TRANSCRIPT string found in message!"

        if "timestamp" not in mess:
            mess["timestamp"] = i  # if no timestamp, at least keep the order
        assert isinstance(mess["timestamp"], int), "timestamp is not int!"

        if "priority" not in mess:
            mess["priority"] = 0
            if "importance" in mess:  # backward compatibility
                mess["priority"] = mess["importance"]
        if isinstance(mess["priority"], float):
            mess["priority"] = int(mess["priority"])
        assert isinstance(mess["priority"], int), f"priority is not int! '{mess['priority']}'"
        assert mess["priority"] <= 10, "priority above 10 !"
        assert mess["priority"] >= -1, "priority under -1 !"

        if "tkn_len" not in mess:
            mess["tkn_len"] = len(tokenize(dedent(mess["content"]).strip()))
            if "answer" in mess:  # system prompt has no answer
                mess["tkn_len"] += len(tokenize(dedent(mess["answer"]).strip()))
            else:
                assert mess["role"] == "system", "expected system message here"
        assert isinstance(mess["tkn_len"], int), "tkn_len is not int!"
        assert mess["tkn_len"] > 0, "tkn_len under 0 !"
        if mess["tkn_len"] > 500 and mess["role"] != "system":
            if mess["priority"] > 5:
                red(f"high priority prompt with more than 500 token: '{mess}'")
            else:
                whi(f"low priority prompt with more than 500 token: '{mess}'")

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


def prompt_filter(prev_prompts, max_token, temperature):
    """goes through the list of previous prompts of the profile, check
    correctness of the key/values, then returns only what's under the maximum
    number of tokens for model"""
    whi("Filtering prompts")

    if temperature != 0:
        whi(f"Temperature is at {temperature}: making the prompt filtering non deterministic.")

    def stocha(n):
        """if temperature of LLM is set high enough, some example filters
        will be randomly discarder to increase randomness. But only after
        the first few prompts were added"""
        if temperature == 0 or n <= 5:
            return True
        threshold = min(temperature / 3, 0.33)
        if random.random() >= threshold:
            # if temp is 1, then 1 in 3 chance of the prompt being ignored by chance
            # no worse if temperature is higher than 1
            return True
        yel(f"Stochasticity decided not to include one prompt (thresh: {threshold:.2f})")
        return False

    assert max_token >= 500, "max_token should be above 500"
    assert max_token <= 15500, "max_token should be under 15500"

    timesorted_pr = sorted(prev_prompts, key=lambda x: x["timestamp"], reverse=True)
    syspr = [pr for pr in prev_prompts if pr["role"] == "system"]
    assert len(syspr) == 1, "Number of system prompts != 1"

    # add by decreasing priority and timestamp
    prio_vals = sorted(set([x["priority"] for x in prev_prompts if int(x["priority"]) != -1]), reverse=True)
    tkns = syspr[0]["tkn_len"]
    dis_tkns = 0
    output_pr = [syspr[0]]
    category_count = 0
    for prio in prio_vals:
        category_size = 0
        for pr in timesorted_pr:
            if pr in output_pr:
                continue
            if pr["priority"] == prio:
                category_size += 1
                if not tkns + pr["tkn_len"] > max_token and stocha(len(output_pr)):
                    tkns += pr["tkn_len"]
                    output_pr.append(pr)
                else:
                    dis_tkns += pr["tkn_len"]
        whi(f"* Keeping {len(output_pr) - category_count} previous prompts that have priority '{prio}' out of {category_size}")  # debug
        category_count = len(output_pr)
    red(f"Tokens of the kept prompts: {tkns} (of all prompts: {tkns + dis_tkns} tokens)")
    yel(f"Total number of prompts saved in memories: '{len(prev_prompts)}'")
    assert len(output_pr) > 1 or len(prev_prompts) == 1, "invalid prompt output"

    # make it so that highest priority prompts are last in the discussion:
    prev_prompts.reverse()
    # or sort by timestamp:
    # prev_prompts = sorted(prev_prompts, key=lambda x: x["timestamp"])
    # make sure the system prompt is first
    for i, p in enumerate(prev_prompts):
        if p["role"] == "system":
            break
    prev_prompts.insert(0, prev_prompts.pop(i))

    assert prev_prompts[0]["role"] == "system", "the first prompt is not system"
    return output_pr


def recur_improv(choice_profile, txt_audio, txt_whisp_prompt, txt_chatgpt_cloz, txt_context, priority):
    whi("Recursively improving")
    if not txt_audio:
        red("No audio transcripts found.")
        return
    if not txt_chatgpt_cloz:
        red("No chatgpt cloze found.")
        return
    if "\n" in txt_chatgpt_cloz:
        whi("Replaced newlines in txt_chatgpt_cloz")
        txt_chatgpt_cloz = txt_chatgpt_cloz.replace("\n", "<br/>")

    content = dedent(transcript_template.replace("CONTEXT", txt_context).replace("TRANSCRIPT", txt_audio)).strip()
    answer = dedent(txt_chatgpt_cloz.replace("\n", "<br/>")).strip()
    tkn_len = len(tokenize(content))
    tkn_len += len(tokenize(answer))
    if tkn_len > 500:
        red("Recursive improvement stopped because you supplied an example "
            f"with a surprising amount of token: '{tkn_len}'")
        return

    prev_prompts = load_prev_prompts(choice_profile)
    try:
        to_add = {
                "role": "user",
                "content": content,
                "timestamp": int(time.time()),
                "priority": priority,
                "answer": answer,
                "llm_model": "gpt-3.5-turbo",
                "tts_model": "whisper-api",
                "tkn_len": tkn_len,
                }
        if to_add in prev_prompts:
            red("Already present in previous outputs!")
            return
        prev_prompts.append(to_add)

        prev_prompts = check_prompts(prev_prompts)

        with open(f"profiles/{choice_profile}/memories.json", "w") as f:
            json.dump(prev_prompts, f, indent=4)
    except Exception as err:
        red(f"Error during recursive improvement: '{err}'")
        return
    whi(f"Recursively improved: {len(prev_prompts)} total examples")


def load_prev_prompts(profile):
    assert Path("profiles/").exists(), "profile directory not found"
    if Path(f"profiles/{profile}/memories.json").exists():
        with open(f"profiles/{profile}/memories.json", "r") as f:
            prev_prompts = json.load(f)
        prev_prompts = check_prompts(prev_prompts)
    else:
        red(f"No memories in profile {profile} found, creating it")
        prev_prompts = check_prompts([default_system_prompt.copy()])
        with open(f"profiles/{profile}/memories.json", "w") as f:
            json.dump(prev_prompts, f, indent=4)

    return prev_prompts
