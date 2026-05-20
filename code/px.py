import webbrowser, urllib.parse

query = "Hello world"#input("Search Google for: ")
url = "https://www.google.com/search?q=" + urllib.parse.quote(query)

webbrowser.open(url)
