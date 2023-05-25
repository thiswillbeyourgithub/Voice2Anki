import hashlib
import json
from pathlib import Path

from .logger import red, whi

pass_salt = "91f0e4cbbf97d6ce8cce29872ac9cacea225ed976a0c1hMiF"


def password_hasher(x):
    def _ph(x):
        return hashlib.sha256(x.encode()).hexdigest()
    hashed_pass = _ph(x + pass_salt)
    for i in range(10_000):
        hashed_pass = _ph(hashed_pass)

def authentification(username, password):
    "simple unsecure authentification with salted multiple hash"

    # checks
    if not (username and password):
        red("No username of password supplied")
        return
    assert isinstance(username, str), "Invalid type of username"
    assert isinstance(password, str), "Invalid type of password"

    with open("users.json", "r") as f:
        user_db = json.load(f)
    if len([k for k in user_db.keys()]) == 0:
        raise Exception "no account in users.json"


    if username not in user_db:
        red(f"Incorrect username '{username}'")
        return False

    # password check
    hashed_pass = password_hasher(password)
    if user_db[username] == hashed_pass:
        return True
    else:
        red(f"Incorrect password for user '{username}'")
        return False

def create_account(username, password):
    "create account and directory"
    whi("Creating account")
    assert username, "empty username"
    assert password, "empty password"
    with open("users.json", "r") as f:
        user_db = json.load(f)
    if username in user_db:
        raise Exception("Username already taken")
    assert not Path(f"../user_data/{username}").exists(), "username folder already exists"
    hashed_pass = password_hasher(password)
    user_db[username] = hashed_pass
    with open("users.json", "w") as f:
        json.dump(user_db, f)
    Path(f"../user_data/{username}").mkdir(exist_ok=False)
    assert Path(f"../user_data/{username}").exists(), "username folder creation failed"
    whi("Done creating account")

def change_password(username, oldpass, newpass):
    whi(f"Changing password")
    assert username, "empty username"
    assert oldpass, "empty old password"
    assert newpass, "empty new password"
    assert oldpass != newpass, "both passwords must be different")
    with open("users.json", "r") as f:
        user_db = json.load(f)
    if username not in user_db:
        raise Exception("Username not found in db")
    hashed_oldpass = password_hasher(oldpass)
    if hashed_oldpass != user_db[username]:
        raise Exception("Old password does not match")
    hashed_newpass = password_hasher(newpass)
    user_db[username] = hashed_newpass
    with open("users.json", "w") as f:
        json.dump(user_db, f)
    whi(f"Done changing password")
    


