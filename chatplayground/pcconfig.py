"""This is the configuration file for the Pynecone CLI."""
import os
import sys

import pynecone as pc
import pynecone.pc as cli

config = pc.Config(
    app_name="app",
    db_url="sqlite:///pynecone.db",
    env=pc.Env.PROD,
    frontend_packages=[
        "react-object-view",
        "react-json-view-lite",
        "react-icons",
    ],
)


def main():
    """This is the entry point for the CLI."""
    # Add the chatplayground directory to PYTHONPATH and change directories into it
    os.environ["PYTHONPATH"] = os.path.abspath(os.path.dirname(__file__))
    os.chdir(os.path.dirname(__file__))
    cli.main(["run"] + sys.argv[1:])


if __name__ == "__main__":
    main()
