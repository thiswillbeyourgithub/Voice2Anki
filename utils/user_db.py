from .logger import red
import hashlib


def authentification(username, password):
    "simple unsecure authentification with salted multiple hash"
    def password_hasher(x):
        return hashlib.sha256(x.encode()).hexdigest()
    user_db = {
            "g": "046db8323e882db07a6171783310495af5f5ebb47b3fd87d4db7afb30e7f2802",
            }
    pass_salt = "91f0e4cbbf97d6ce8cce29872ac9cacea225ed976a0c1hMiF"

    # checks
    if not (username and password):
        red("No username of password supplied")
        return
    assert isinstance(username, str), "Invalid type of username"
    assert isinstance(password, str), "Invalid type of password"
    if username not in user_db:
        red(f"Incorrect username '{username}'")
        return False

    # password check
    hashed_pass = password_hasher(password + pass_salt)
    for i in range(10_000):
        hashed_pass = password_hasher(hashed_pass)
    if user_db[username] == hashed_pass:
        return True
    else:
        red(f"Incorrect password for user '{username}'")
        return False

