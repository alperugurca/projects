# Remove extra Spaces from the data
text = "Hello,    World!    2035"
text.split()

"""
text.split()
['Hello,', 'World!', '2035']
"""

cleaned_text1=" ".join(text.split())
print(f"text: {text} \n cleaned_text1: {cleaned_text1}")
# %% Upper to lower case
text = "Hello, World! 2035"
cleaned_text2 = text.lower() # make lower case
print(f"text: {text} \n cleaned_text2: {cleaned_text2}")
# %% Remove punctuation
import string
text = "Hello, World! 2035"
cleaned_text3 = text.translate(str.maketrans("", "", string.punctuation))
print(f"text: {text} \n cleaned_text3: {cleaned_text3}")
# %% Remove special characters, %, @, /, *, #
import re
text = "Hello, World! 2035%"
cleaned_text4 = re.sub(r"[^A-Za-z0-9\s]", "", text)
print(f"text: {text} \n cleaned_text4: {cleaned_text4}")

# %% Correct spelling
from textblob import TextBlob # Text analysis library

text = "Hellio, Wirld! 2035"
cleaned_text5 = TextBlob(text).correct()
print(f"text: {text} \n cleaned_text5: {cleaned_text5}")

# %% html or url tags remove
from bs4 import BeautifulSoup

html_text = "<div>Hello, World! 2035</div>"

cleaned_text6 = BeautifulSoup(html_text, "html.parser").get_text()

print(f"html_text: {html_text} \n cleaned_text6: {cleaned_text6}")
