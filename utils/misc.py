import ftfy

# string at the end of the prompt
prompt_finish = "\n\n###\n\n"

# string at the end of the completion
completion_finish = "\n END"

# characters that separate the cloze in the completion of the model
cloze_completion_separator = "\n'''\n"

def convert_paste(paste):
    try:
        print(f"TODO: fix this.")
        # is text
        return ftfy.fix_text(str(paste))
    except:
        # is image
        return paste
