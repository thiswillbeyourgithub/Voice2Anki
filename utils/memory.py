from tqdm import tqdm
import numpy as np
import random
import time
from pathlib import Path
from textwrap import dedent
import json
import hashlib
from joblib import Memory
from sentence_transformers import SentenceTransformer, util
import tiktoken

from .logger import whi, red, yel, trace, Timeout
from .shared_module import shared

# string at the end of the prompt
prompt_finish = "\n\n###\n\n"

# string at the end of the completion
completion_finish = "\n END"

# used to count the number of tokens for chatgpt
tokenizer = tiktoken.encoding_for_model("gpt-3.5-turbo")
tokenize = tokenizer.encode

transcript_template = """
Context: 'CONTEXT'
Transcript: 'TRANSCRIPT'
""".strip()

default_system_prompt_md = {
             "role": "system",
             "content": dedent("""You are my excellent assistant Alfred. You always exceed my expectations. Your task today is the to transform audio transcripts into markdown formatted text.

             You follow these rules:
             - always end your replies by "END".
             - separate bullet points by a dash '-'
             - the transcript can be of poor quality, it is your job to correct transcription errors using the context.
             - If relevant, use LaTeX formatting in your answer.
             - if you're absolutely certain that you can't accomplish your task: begin your answer by 'Alfred:' and I'll take a look immediately."""),
             "timestamp": int(time.time()),
             "priority": -1,  # the only prompt that has priority of -1 is the system prompt
             }


default_system_prompt_anki = {
            "role": "system",
            "content": dedent("""You are my excellent assistant Alfred. You always exceed my expectations. Your task today is the to transform audio transcripts into Anki flashcards.

            You follow these rules:
            - assume the information in the card is true.
            - always reuse acronyms from the transcript.
            - always end your replies by "END".
            - If you create several flashcards for one transcript, separate them with "#####".
            - always correct transcription mistakes.
            - if you're absolutely certain that you can't accomplish your task: begin your answer by 'Alfred:' and I'll take a look immediately."""),
            "timestamp": int(time.time()),
            "priority": -1,  # the only prompt that has priority of -1 is the system prompt
            }

if shared.backend == "anki":
    default_system_prompt = default_system_prompt_anki
    backend = "anki"
elif shared.backend == "markdown":
    default_system_prompt = default_system_prompt_md
    backend = "markdown"
else:
    raise Exception(shared.backend)

expected_mess_keys = ["role", "content", "timestamp", "priority", "tkn_len_in", "tkn_len_out", "answer", "llm_model", "tts_model", "hash"]

embedding_model_name = "paraphrase-multilingual-MiniLM-L12-v2"
embeddings_cache = Memory(f".cache/{embedding_model_name}", verbose=0)
embed_model = SentenceTransformer(embedding_model_name)


@embeddings_cache.cache
def embedder(text):
    red("Computing embedding of 1 memory")
    # remove the context before the transcript as well as the last '
    text = text.split("Transcript: '")
    if not len(text) == 2:
        raise Exception(text)
    text = text[1]
    if not text[-1] == "'":
        raise Exception(text)
    text = text[:-1]

    return embed_model.encode([text], show_progress_bar=False).tolist()[0]

def hasher(text):
    return hashlib.sha256(text.encode()).hexdigest()[:10]

@trace
def check_prompts(prev_prompts):
    "checks validity of the previous prompts"
    whi("Checking prompt validity")
    for i, mess in enumerate(prev_prompts):

        if mess["role"] == "user":
            assert "answer" in mess, "no answer key in message"
            assert "{{c1::" in mess["answer"], f"No cloze found in {mess}"
            assert "}}" in mess["answer"], f"No cloze found in {mess}"
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
            if k not in keys:
                del prev_prompts[i][k]

        # make sure it's stripped
        prev_prompts[i]["content"] = dedent(prev_prompts[i]["content"]).strip()
        if "answer" in prev_prompts[i]:
            prev_prompts[i]["answer"] = dedent(prev_prompts[i]["answer"]).strip()

    return prev_prompts


def filter_out(pr, tkns, output_pr, max_token, temperature, favor_list, new_prompt_len, sig, dist_check):
    """apply a list of criteria to keep the most relevant previous memories
    in the prompt"""
    if tkns + pr["tkn_len_in"] + pr["tkn_len_out"] > max_token:
        return False

    if not favor_list:  # the txt_audio does not ask for a list
        if " list" in pr["content"].lower():
            # exclude list cards if not asking for a list
            return False

        # semantic similarity check
        if dist_check == 0:
            # ignored because is in the cards with the lowest similarity
            return False

        # length check
        if not abs(np.log(pr["tkn_len_in"]) - np.log(new_prompt_len)) <= 2 * sig:
            # whi(f"Rejected prompt: pl {new_prompt_len}, sig {np.exp(sig)}, tknlen {pr['tkn_len_in']}")
            return False

        # stochastic check
        if temperature > 0.3:
            # if temperature of LLM is set high enough, some example filters
            # will be randomly discarder to increase randomness. But only after
            # the first few prompts were added
            threshold = min(temperature / 3, 0.33)
            if random.random() < threshold:
                # if temp is 1, then 1 in 3 chance of the prompt being ignored by chance
                # no worse if temperature is higher than 1
                return False

        # passed all tests
        return True

    else:  # if favoring lists, don't use stochastic check
        # candidate is not a list
        if not "list" in pr["content"].lower():
            return False

        # semantic similarity check
        if dist_check == 0:
            # ignored because is in the cards with the lowest similarity
            return False

        return True


@trace
@Timeout(30)
def prompt_filter(prev_prompts, max_token, temperature, new_prompt_len, new_prompt_vec, favor_list):
    """goes through the list of previous prompts of the profile, check
    correctness of the key/values, then returns only what's under the maximum
    number of tokens for model"""
    whi("Filtering prompts")

    if temperature != 0:
        whi(f"Temperature is at {temperature}: making the prompt filtering non deterministic.")


    assert max_token >= 500, "max_token should be above 500"
    assert max_token <= 15500, "max_token should be under 15500"

    # get average and spread of tkns lengths to keep only the most similar
    lens = [p["tkn_len_in"] for p in prev_prompts]
    llens = np.log(lens)
    sig = np.std(llens)

    timesorted_pr = sorted(prev_prompts, key=lambda x: x["timestamp"], reverse=True)
    syspr = [pr for pr in prev_prompts if pr["role"] == "system"]
    assert len(syspr) == 1, "Number of system prompts != 1"
    assert syspr[0] == timesorted_pr[-1], "System prompt is not the oldest memory!"

    if shared.disable_embeddings:
        whi("Not using embeddings")
        dist_check = [1 for i in timesorted_pr[:-1]] + [0]
    else:
        # the system prompt is the oldest and is not embedder
        embeddings_content = [embedder(pr["content"]) for pr in tqdm(timesorted_pr[:-1], desc="computing embeddings")] + [None]
        # embeddings_answer = [embedder(pr["answer"]) for pr in timesorted_pr[:-1]] + [None]

        whi("Computing cosine similarity")
        distances = []
        max_sim = [0, None]
        min_sim = [1, None]
        for i in range(len(timesorted_pr)):
            if embeddings_content[i] is None:  # system_prompt
                content_dist = 0
                # answer_dist = 0
            else:
                content_dist = float(util.cos_sim(new_prompt_vec, embeddings_content[i]))
                # answer_dist = float(util.cos_sim(new_prompt_vec, embeddings_answer[i]))
            score = content_dist * 1
            # score += answer_dist * 1
            distances.append(score)
            if score > max_sim[0]:
                max_sim[0] = score
                max_sim[1] = timesorted_pr[i]
            if score < min_sim[0]:
                min_sim[0] = score
                min_sim[1] = timesorted_pr[i]
        red(f"Memory with lowest similarity is: '{min_sim}'")
        red(f"Memory with highest similarity is: '{max_sim}'")

        plimit = 90
        percentile = float(np.percentile(distances, plimit))
        dist_check = [1 if d >= percentile else 0 for d in distances]
    assert len(dist_check) == len(timesorted_pr), "unexpected length"

    # add by decreasing priority and timestamp
    prio_vals = sorted(set([x["priority"] for x in prev_prompts if int(x["priority"]) != -1]), reverse=True)
    tkns = syspr[0]["tkn_len_in"]
    tkns += new_prompt_len
    dis_tkns = 0
    output_pr = [syspr[0]]  # add system prompt

    # add automatically the highest similarity
    if not shared.disable_embeddings:
        output_pr.append(max_sim[1])
        tkns += max_sim[1]["tkn_len_in"]

    exit_while = False
    cnt = 0
    while (not exit_while) and timesorted_pr:
        category_count = 0
        cnt += 1
        if cnt > 20:
            red(f"Exited filtering loop after {cnt} iterations, have you added enough memories?")
            exit_while = True
            break
        for prio in prio_vals:
            category_size = 0
            if exit_while:
                break
            for pr_idx, pr in enumerate(timesorted_pr):
                if pr in output_pr:
                    continue
                if pr["priority"] == prio:
                    category_size += 1

                    if tkns + pr["tkn_len_in"] + pr["tkn_len_out"] > max_token:
                        # will exit while at the end of this loop but not
                        # before
                        exit_while = True

                    if filter_out(
                            pr,
                            tkns,
                            output_pr,
                            max_token,
                            temperature,
                            favor_list,
                            new_prompt_len,
                            sig,
                            dist_check[pr_idx],
                            ):
                        tkns += pr["tkn_len_in"] + pr["tkn_len_out"]
                        output_pr.append(pr)
                    else:
                        dis_tkns += pr["tkn_len_in"] + pr["tkn_len_out"]
            whi(f"* {cnt} Keeping {len(output_pr) - category_count} previous prompts that have priority '{prio}' out of {category_size}")  # debug
            category_count = len(output_pr)

        red(f"Finished looping over all the memories with only {len(output_pr)} prompts selected, so relaxing the length limit")
        sig -= sig * 0.1
        sig = max(sig, 0)
        if not shared.disable_embeddings:
            plimit -= 10
            plimit = max(plimit, 0)
            percentile = float(np.percentile(distances, plimit))
            dist_check = [1 if d >= percentile else 0 for d in distances]

    red(f"Tokens of the kept prompts: {tkns} (of all prompts: {tkns + dis_tkns} tokens)")
    yel(f"Total number of prompts saved in memories: '{len(prev_prompts)}'")

    # # make it so that highest priority prompts are last in the discussion:
    # output_pr.reverse()

    if not shared.disable_embeddings:
        # sort by distance:
        output_pr = sorted(output_pr, key=lambda x: distances[output_pr.index(x)])
    else:
        # or by timestamp (most recent last):
        output_pr = sorted(output_pr, key=lambda x: x["timestamp"])
        # or by priority:
        # output_pr = sorted(output_pr, key=lambda x: x["priority"])

    # regardless, make sure the system prompt is first
    for i, p in enumerate(output_pr):
        if p["role"] == "system":
            break
    output_pr.insert(0, output_pr.pop(i))
    assert output_pr[0]["role"] == "system", "the first prompt is not system"

    return output_pr


@trace
def recur_improv(txt_profile, txt_audio, txt_whisp_prompt, txt_chatgpt_outputstr, txt_context, priority, check_gpt4):
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
                "llm_model": shared.latest_llm_used,
                "tts_model": shared.latest_stt_used,
                "tkn_len_in": tkn_len_in,
                "tkn_len_out": tkn_len_out,
                "hash": hasher(content),
                }
        if to_add["hash"] in [pp["hash"] for pp in prev_prompts]:
            red("This prompt is already present in the memory!")
            return
        prev_prompts.append(to_add)

        prev_prompts = check_prompts(prev_prompts)

        with open(f"profiles/{backend}/{txt_profile}/memories.json", "w") as f:
            json.dump(prev_prompts, f, indent=4, ensure_ascii=False)
    except Exception as err:
        red(f"Error during recursive improvement: '{err}'")
        return
    whi(f"Recursively improved: {len(prev_prompts)} total examples")


@trace
def load_prev_prompts(profile):
    assert Path("profiles/").exists(), "profile directory not found"
    if Path(f"profiles/{backend}/{profile}/memories.json").exists():
        with open(f"profiles/{backend}/{profile}/memories.json", "r") as f:
            prev_prompts = json.load(f)
        prev_prompts = check_prompts(prev_prompts)
    else:
        red(f"No memories in profile {profile} found, creating it")
        prev_prompts = check_prompts([default_system_prompt.copy()])
        with open(f"profiles/{backend}/{profile}/memories.json", "w") as f:
            json.dump(prev_prompts, f, indent=4, ensure_ascii=False)

    return prev_prompts


def display_price(sld_max_tkn, check_gpt4):
    if check_gpt4:
        price = [0.03, 0.06]
    else:
        price = [0.001, 0.002]
    price_adj = price[0] * 0.9 + price[1] * 0.1
    price_per_request = price_adj * sld_max_tkn / 1000
    price_per_dol = round(1 / price_per_request, 0)
    message = f"Price if all tokens used: ${price_per_request:.5f}"
    message += f"\nRequests per $1: {price_per_dol:.1f} req"
    return message

def show_memories(profile):
    memories = load_prev_prompts(profile)
    output = [""]
    for memory in memories:
        for k, v in memory.items():
            output[-1] += f"{k.upper()}: {v}\n"
        output.append("")
    return "\n\n".join(output[:-1])
