"""Frozen entry point for the personal local authentication server."""

import multiprocessing

import uvicorn

from Server import app


def main():
    multiprocessing.freeze_support()
    uvicorn.run(app, host='127.0.0.1', port=8000, log_level='info')


if __name__ == '__main__':
    main()
