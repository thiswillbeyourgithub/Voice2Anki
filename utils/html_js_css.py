from .shared_module import shared

darkmode_js = """() => {
    if (document.querySelectorAll('.dark').length) {
        document.querySelectorAll('.dark').forEach(el => el.classList.remove('dark'));
    } else {
        document.querySelector('body').classList.add('dark');
    }
}"""

html_head = """
<script>

function unfocus(e) {
    document.activeElement.blur();
    document.body.focus();
    document.documentElement.focus();
    e.preventDefault();  // make sure to avoid scrolling after pressing spacebar
}

function shortcuts(e) {
    // the switch is so that keypress are ignored if an input element is
    // in focus
    var event = document.all ? window.event : e;
    switch (e.target.tagName.toLowerCase()) {

        // unselect anything by pressing shift+space or escape
        case "input":
        case "textarea":
            if ((e.code == 'Space' && e.shiftKey) || (e.key == 'Escape') || (e.keyCode == 27)) {
                unfocus(e);
            }

        //case "select":
        //case "button":
        break;

        default:
        // suspend previous card
        if (e.code == "KeyS" && e.shiftKey) {
            if (!document.querySelector('.js_tabmain').checkVisibility()) { alert("Shortcut only available in tab 'Main'") ; return ;}
            document.getElementById("js_suspendpreviousbtn").click();
        }
        // mark previous card
        else if (e.code == 'Semicolon' && e.shiftKey) {
            if (!document.querySelector('.js_tabmain').checkVisibility()) { alert("Shortcut only available in tab 'Main'") ; return ;}
            document.getElementById("js_markpreviousbtn").click();
        }
        // untoggle check next card
        else if (e.key == 'm') {
            if (!document.querySelector('.js_tabmain').checkVisibility()) { alert("Shortcut only available in tab 'Main'") ; return ;}
            document.getElementById("js_marknext").children[1].children[0].checked = false;
        }
        // get card status
        else if (e.key == "s") {
            if (!document.querySelector('.js_tabmain').checkVisibility()) { alert("Shortcut only available in tab 'Main'") ; return ;}
            document.getElementById("js_cardstatusbtn").click();
        }
        // recur improvement
        else if (e.code == "KeyF" && e.shiftKey) {
            if (!document.querySelector('.js_tabmain').checkVisibility()) { alert("Shortcut only available in tab 'Main'") ; return ;}
            document.getElementById("js_llmfeedbackbtn").click();
        }

        // toggle nightmode
        else if (e.code == "KeyN" && e.shiftKey) {
            document.getElementById("js_darkmodebtn").click();
        }

        // select textbox
        else if (e.key == "e") {
            e.preventDefault();  // dont type the e
            if (!document.querySelector('.js_tabmain').checkVisibility()) { alert("Shortcut only available in tab 'Main'") ; return ;}
            ch = document.getElementById("js_txtchatgpt").children;
            ch[ch.length - 1].focus();
        }
        else if (e.code == "KeyE" && e.shiftKey) {
            e.preventDefault();  // dont type the e
            if (!document.querySelector('.js_tabmain').checkVisibility()) { alert("Shortcut only available in tab 'Main'") ; return ;}
            ch = document.getElementById("js_txtwhisper").children;
            ch[ch.length - 1].focus();
        }

        // roll 1 2 3
        else if (e.key == "&") {
            if (!document.querySelector('.js_tabmain').checkVisibility()) { alert("Shortcut only available in tab 'Main'") ; return ;}
            document.getElementById("js_roll1").click();
        }
        else if (e.key == "é") {
            if (!document.querySelector('.js_tabmain').checkVisibility()) { alert("Shortcut only available in tab 'Main'") ; return ;}
            document.getElementById("js_roll12").click();
        }
        else if (e.key == '"') {
            if (!document.querySelector('.js_tabmain').checkVisibility()) { alert("Shortcut only available in tab 'Main'") ; return ;}
            document.getElementById("js_roll123").click();
        }

        // 123
        else if (e.key == "3" && e.shiftKey) {
            if (!document.querySelector('.js_tabmain').checkVisibility()) { alert("Shortcut only available in tab 'Main'") ; return ;}
            document.getElementById("js_toankibtn").click();
        }
        else if (e.key == "2" && e.shiftKey) {
            if (!document.querySelector('.js_tabmain').checkVisibility()) { alert("Shortcut only available in tab 'Main'") ; return ;}
            document.getElementById("js_transcriptbtn").click();
        }
        else if (e.key == "1" && e.shiftKey) {
            if (!document.querySelector('.js_tabmain').checkVisibility()) { alert("Shortcut only available in tab 'Main'") ; return ;}
            document.getElementById("js_transcribebtn").click();
        }

        // roll gallery
        else if (document.getElementById('js_guienablequeuedgallery').children[1].children[0].checked == true && e.code == 'KeyG' && e.shiftKey) {
            if (!document.querySelector('.js_tabmain').checkVisibility()) { alert("Shortcut only available in tab 'Main'") ; return ;}
            if (confirm("Roll gallery?")) {
                document.getElementById("js_rollgallbtn").click();
            }
        }
        // add to next queued gallery
        else if (document.getElementById('js_guienabledirload').children[1].children[0].checked == true && e.code == 'KeyR' && e.shiftKey) {
            if (!document.querySelector('.js_tabqueues').checkVisibility()) { alert("Shortcut only available in tab 'Queues'") ; return ;}
            // only active if the right tabs are enabled
            if (document.querySelector('.js_tabqueues').checkVisibility() && document.querySelector('.js_queueqgclass').checkVisibility()) {
                    document.getElementById("js_btnqgnew").click();
            }
            else {
                alert("To add to queued gallery, go to 'Queues' then 'Queud audio'");
            }
        }
        // append to latest queueud gallery
        else if (document.getElementById('js_guienabledirload').children[1].children[0].checked == true && e.code == 'KeyQ' && e.shiftKey) {
            if (!document.querySelector('.js_tabqueues').checkVisibility()) { alert("Shortcut only available in tab 'Queues'") ; return ;}
            // only active if the right tabs are enabled
            if (document.querySelector('.js_tabqueues').checkVisibility() && document.querySelector('.js_queueqgclass').checkVisibility()) {
                document.getElementById("js_btnqgadd").click();
            }
            else {
                alert("To add to queued gallery, go to 'Queues' then 'Queud audio'");
            }
        }

        // dirload
        else if (document.getElementById('js_guienabledirload').children[1].children[0].checked == true && e.code == 'KeyD' && e.shiftKey) {
            if (!document.querySelector('.js_tabmain').checkVisibility()) { alert("Shortcut only available in tab 'Main'") ; return ;}
            if (confirm("Load from dir?")) {
                document.getElementById("js_dirloadbtn").click();
            }
        }

        // switch top tabs
        else if (e.code == "KeyT") {
            var tabs = document.querySelectorAll(".js_toptabs")[0].parentElement.childNodes[0].children;
            Array.from(tabs).forEach((tab, index) => {
                if (tab.ariaSelected == 'true') {
                    if (event.shiftKey) {
                        var newIndex = (index > 0) ? index - 1 : tabs.length - 1;
                    } else {
                        var newIndex = (index + 1) % tabs.length;
                    }
                    tabs[newIndex].click();
                    return;
                }
            });
        }

        // unfocus
        else if ((e.code == 'Space' && e.shiftKey) || (e.key == 'Escape') || (e.keyCode == 27)) {
            unfocus(e);
        }

        // ignore space
        else if ((e.code == 'Space' && e.shiftKey) || (e.code == 'Space')) {
            // do nothing
        }

        // no shortcut found
        else {
            alert(`Unrecognized shortcut: ${e.key} (or ${e.code})`);
            }

        }
}
function tabswitcher(e) {
    // tab to switch tab
    if (event.key === 'Tab') {
        event.preventDefault();

        // the subtabs cycled depend on the main tab focused
        if (document.querySelector('.js_tabmain').checkVisibility()) {
            var selector = '.js_subtab_main'
        }
        else if (document.querySelector('.js_tabsettings').checkVisibility()) {
            var selector = '.js_subtab_settings'
        }
        else if (document.querySelector('.js_tabqueues').checkVisibility()) {
            var selector = '.js_subtab_queues'
        }
        else if (document.querySelector('.js_tabmemoriesandbuffer').checkVisibility()) {
            var selector = '.js_subtab_memoriesandbuffer'
        }
        else {
            // alert("No subtab to switch here.");
            return;
        }


        var tabs = document.querySelectorAll(selector)[0].parentElement.childNodes[0].children;
        Array.from(tabs).forEach((tab, index) => {
            if (tab.ariaSelected == 'true') {
                if (event.shiftKey) {
                    var newIndex = (index > 0) ? index - 1 : tabs.length - 1;
                } else {
                    var newIndex = (index + 1) % tabs.length;
                }
                tabs[newIndex].click();
                return;
            }
        });

    }
}

document.addEventListener('keypress', shortcuts, false);
document.addEventListener('keydown', tabswitcher, false);


////// code related to syntax highlighting
//const rules = [
//  { regex: /\b(for)\b/g, replacement: '<span style="color: red;">$1</span>' },
//  { regex: /\b(if)\b/g, replacement: '<span style="color: blue;">$1</span>' },
//  { regex: /\b(else)\b/g, replacement: '<span style="color: green;">$1</span>' },
//];
//// Apply syntax highlighting
//function applySyntaxHighlighting(html) {
//  rules.forEach(rule => {
//    html = html.replace(rule.regex, rule.replacement);
//  });
//  return html;
//}
//// Function to preserve caret position
//function getCaretPosition(editableDiv) {
//  let caretPos = 0, sel, range;
//  if (window.getSelection) {
//    sel = window.getSelection();
//    if (sel.rangeCount) {
//      range = sel.getRangeAt(0);
//      if (range.commonAncestorContainer === editableDiv.parentElement) {
//        caretPos = range.endOffset;
//      }
//    }
//  }
//  return caretPos;
//}
//// Set caret position
//function setCaretPosition(editableDiv, position) {
//  if (window.getSelection && document.createRange) {
//    const range = document.createRange();
//    range.selectNodeContents(editableDiv.parentElement);
//    range.collapse(true);
//    range.setStart(editableDiv.parentElement, position);
//    range.setEnd(editableDiv.parentElement, position);
//    const sel = window.getSelection();
//    sel.removeAllRanges();
//    sel.addRange(range);
//  }
//}
//// Main function to handle input and styling
//function handleInput(event) {
//  const target = event.target;
//  const caretPosition = getCaretPosition(target);
//  let content = target.innerText;
//  target.innerHTML = applySyntaxHighlighting(content);
//  setCaretPosition(target, caretPosition);
//}
//
//el=document.getElementById("js_txtchatgpt").childNodes[1].childNodes[5];
//el.contentEditable = true;
//el.addEventListener('input', handleInput);


//// code to create a text input area on top of txtchatgpt:
//// Assuming you have an existing element with id 'existingElement'
//const existingElement = document.getElementById('js_txtchatgpt');
//
//// Create new text input element
//const newTextElement = document.createElement('input');
//newTextElement.type = 'text';
//
//// Get coordinates and dimensions of existingElement
//const rect = existingElement.childNodes[1].childNodes[5].getBoundingClientRect();
//
//// Set style of newTextElement for exact overlay based on viewport position
//newTextElement.style.position = 'absolute';
//newTextElement.style.top = `${rect.top + window.scrollY}px`; // Adjust for scrolling
//newTextElement.style.left = `${rect.left + window.scrollX}px`; // Adjust for scrolling
//newTextElement.style.width = `${rect.width}px`;
//newTextElement.style.height = `${rect.height}px`;
//
//// Link contents
//newTextElement.oninput = () => existingElement.childNodes[1].childNodes[5].value = newTextElement.value;
////existingElement.oninput = () => newTextElement.value = existingElement.childNodes[1].childNodes[5].value;
//
//// Append newTextElement to body to ensure it is positioned based on viewport coordinates
//document.body.appendChild(newTextElement);


</script>
"""

# dynamically adjust the height of the app to avoid scrolling up abruptly
js_longer = """() => {
    document.querySelectorAll(".app")[0].style.height='5000px';
}
"""
js_reset_height = """() => {
    document.querySelectorAll(".app")[0].style.height='';
}
"""

# executed on load
js_load = """() => {
    // make sure the audios keep the same size even when they are unset
    var h = Math.max(90, Math.floor(2.3 * document.getElementsByClassName("js_audiocomponent")[0].clientHeight));


    Array.from(document.getElementsByClassName("js_audiocomponent")).forEach(el => el.style.height = `${h}px`)

}
"""

css = """
/* make sure those tabs take all the width */
#js_widetabs-button { flex-grow: 1 !important;}

/* remove source selector */
.mic-select {display: none !important; flex-grow:0 !important;}
"""

if shared.big_font:
    css += """
/* Larger font for some text elements */
#js_txtchatgpt > label > textarea {font-size: 20px;}
#js_txtwhisper > label > textarea {font-size: 20px;}
"""
# else:
#     css += """
# /* Larger font for some text elements */
# #js_txtchatgpt > label > textarea {font-size: 17px;}
# #js_txtwhisper > label > textarea {font-size: 17px;}
# """

if shared.widen_screen:
    css += "\n.app { max-width: 100% !important; }"
    css += "\n.app { width: 100% !important; }"

