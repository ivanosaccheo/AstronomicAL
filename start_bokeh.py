
import os
import sys
import platform
import webbrowser

if platform.system() == 'Linux':
    import multiprocessing
    multiprocessing.set_start_method("fork")
    print(">>> Forking enabled for Linux.")

print(">>> Starting the Bokeh server...")

from bokeh.server.server import Server
from bokeh.application import Application
from bokeh.application.handlers.directory import DirectoryHandler

# Update the path to your directory (astronomicAL)
apps = {
    "/": Application(DirectoryHandler(filename="astronomicAL"))
}

server = Server(apps, port=5006)
print(">>> Server started on http://localhost:5006")

webbrowser.open("http://localhost:5006")

server.start()
server.io_loop.start()