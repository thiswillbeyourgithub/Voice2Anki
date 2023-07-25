import sys
import numpy as np
import random
import time
from pathlib import Path
from textwrap import dedent
import json
import hashlib

from .logger import whi, red, yel
from .misc import tokenize, transcript_template

global prev_prompts

default_system_prompt_md = {
             "role": "system",
             "content": dedent("""You are my excellent assistant Alfred. You always exceed my expectations. Your task today is the to transform audio transcripts into markdown formatted text.

             Here are the rules:
             - always end your replies by "END".
             - separate bullet points by '-'
             - the transcript can be of poor quality, it is your job to correct transcription errors using the context.
             - If relevant, use LaTeX formatting in your answer.
             - if you're absolutely certain that you can't accomplish your task: begin your answer by 'Alfred:' and I'll take a look immediately."""),
             "timestamp": int(time.time()),
             "priority": -1,  # the only prompt that has priority of -1 is the system prompt
             }


default_system_prompt_anki = {
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

args = sys.argv[1:]
if "--backend=anki" in args:
    default_system_prompt = default_system_prompt_anki
elif "--backend=markdown" in args:
    default_system_prompt = default_system_prompt_md
else:
    raise Exception

expected_mess_keys = ["role", "content", "timestamp", "priority", "tkn_len_in", "tkn_len_out", "answer", "llm_model", "tts_model", "hash"]

def hasher(text):
    return hashlib.sha256(text.encode()).hexdigest()[:10]

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

        if "hash" not in mess:
            mess["hash"] = hasher(mess["content"])

        if "tkn_len_in" not in mess:
            mess["tkn_len_in"] = len(tokenize(dedent(mess["content"]).strip()))
            if "answer" in mess:  # system prompt has no answer
                mess["tkn_len_out"] = len(tokenize(dedent(mess["answer"]).strip()))
            else:
                assert mess["role"] == "system", "expected system message here"
                mess["tkn_len_out"] = 0
        assert isinstance(mess["tkn_len_in"], int), "tkn_len_in is not int!"
        assert isinstance(mess["tkn_len_out"], int), "tkn_len_out is not int!"
        assert mess["tkn_len_in"] > 0, "tkn_len_in under 0 !"
        assert mess["tkn_len_out"] > 0 or mess["role"] == "system", "tkn_len_out under 0 !"
        if mess["tkn_len_in"] + mess["tkn_len_out"] > 500:
            if mess["priority"] > 5:
                red(f"high priority prompt with more than 500 token: '{mess}'")
            else:
                whi(f"low priority prompt with more than 500 token: '{mess}'")

        # keep only the expected keys
        keys = [k for k in mess.keys() if k in expected_mess_keys]
        for k in prev_prompts[i]:
            if k not in prev_prompts[i]:
                del prev_prompts[i][k]

        # make sure it's stripped
        prev_prompts[i]["content"] = dedent(prev_prompts[i]["content"]).strip()
        if "answer" in prev_prompts[i]:
            prev_prompts[i]["answer"] = dedent(prev_prompts[i]["answer"]).strip()

    return prev_prompts


def prompt_filter(prev_prompts, max_token, temperature, new_prompt_len=None):
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

    if new_prompt_len:
        # get average and spread of tkns lengths:
        lens = [p["tkn_len_in"] for p in prev_prompts]
        llens = np.log(lens)
        sig = np.std(llens)

        def len_check(pr):
            if abs(np.log(pr["tkn_len_in"]) - np.log(new_prompt_len)) <= 2 * sig:
                red(f"Accepted prompt: pl {new_prompt_len}, sig {np.exp(sig)}, tknlen {pr['tkn_len_in']}")
                return True
            else:
                yel(f"Rejected prompt: pl {new_prompt_len}, sig {np.exp(sig)}, tknlen {pr['tkn_len_in']}")
                return False
    else:
        def len_check(pr):
            return True


    timesorted_pr = sorted(prev_prompts, key=lambda x: x["timestamp"], reverse=True)
    syspr = [pr for pr in prev_prompts if pr["role"] == "system"]
    assert len(syspr) == 1, "Number of system prompts != 1"

    # add by decreasing priority and timestamp
    prio_vals = sorted(set([x["priority"] for x in prev_prompts if int(x["priority"]) != -1]), reverse=True)
    tkns = syspr[0]["tkn_len_in"]
    dis_tkns = 0
    output_pr = [syspr[0]]  # add system prompt
    category_count = 0
    for prio in prio_vals:
        category_size = 0
        for pr in timesorted_pr:
            if pr in output_pr:
                continue
            if pr["priority"] == prio:
                category_size += 1
                if not tkns + pr["tkn_len_in"] + pr["tkn_len_out"] > max_token and stocha(len(output_pr)) and len_check(pr):
                    tkns += pr["tkn_len_in"] + pr["tkn_len_out"]
                    output_pr.append(pr)
                else:
                    dis_tkns += pr["tkn_len_in"] + pr["tkn_len_out"]
        whi(f"* Keeping {len(output_pr) - category_count} previous prompts that have priority '{prio}' out of {category_size}")  # debug
        category_count = len(output_pr)
    red(f"Tokens of the kept prompts: {tkns} (of all prompts: {tkns + dis_tkns} tokens)")
    yel(f"Total number of prompts saved in memories: '{len(prev_prompts)}'")

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


def recur_improv(txt_profile, txt_audio, txt_whisp_prompt, txt_chatgpt_outputstr, txt_context, priority):
    whi("Recursively improving")
    if not txt_audio:
        red("No audio transcripts found.")
        return
    if not txt_chatgpt_outputstr:
        red("No chatgpt output string found.")
        return
    if "\n" in txt_chatgpt_outputstr:
        whi("Replaced newlines in txt_chatgpt_outputstr")
        txt_chatgpt_outputstr = txt_chatgpt_outputstr.replace("\n", "<br/>")

    content = dedent(transcript_template.replace("CONTEXT", txt_context).replace("TRANSCRIPT", txt_audio)).strip()
    answer = dedent(txt_chatgpt_outputstr.replace("\n", "<br/>")).strip()
    tkn_len_in = len(tokenize(content))
    tkn_len_out = len(tokenize(answer))
    if tkn_len_in + tkn_len_out > 500:
        red("You supplied an example "
            f"with a surprising amount of token: '{tkn_len_in + tkn_len_out}' This can have "
            "adverse effects.")

    prev_prompts = load_prev_prompts(txt_profile)
    try:
        to_add = {
                "role": "user",
                "content": content,
                "timestamp": int(time.time()),
                "priority": priority,
                "answer": answer,
                "llm_model": "gpt-3.5-turbo-4k_or_16k",
                "tts_model": "whisper-api",
                "tkn_len_in": tkn_len_in,
                "tkn_len_out": tkn_len_out,
                "hash": hasher(content),
                }
        if to_add["hash"] in [pp["hash"] for pp in prev_prompts]:
            red("This prompt is already present in the memory!")
            return
        prev_prompts.append(to_add)

        prev_prompts = check_prompts(prev_prompts)

        with open(f"profiles/{txt_profile}/memories.json", "w") as f:
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
