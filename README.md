# Powerplant Coding Challenge

## How to run

First make sure [poetry](https://python-poetry.org/docs/#installation) is installed. Then, within the main folder, run 
    `poetry install`
on a command line to install the virtual environment with the proper dependencies. Finally, run
    `poetry run uvicorn app:app --port 8888`
to start the fastAPI webserver. It should now provide a proper response to a POST request on http://localhost:8888/powerplants. Perhaps more conveniently, you can also interact with the API through visiting http://localhost:8888/docs in your browser.
