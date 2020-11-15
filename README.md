# Experiment-2020-FutureInternet-LoRa

## Experimental results

This repository contains code and instruction to reproduce the experiments presented in the paper "Deployment Independence for Self-Organising Cyber-Physical Systems" by Roberto Casadei, Danilo Pianini, Andrea Placuzzi, Mirko Viroli, and Danny Weyns; submitted to the journal Future Internet.

## Requirements

In order to run the experiments, the Java Development Kit 11 is required.
We use a version of the DingNet simulator compatible with the JDK up to version 14.
We use a version of the Protelis language compatible with the JDK up to version 13.
We test using OpenJDK 11 and 14.

In order to produce the charts, Python 3 is required.
We recommend Python 3.8.1,
but it is very likely that any Python 3 version,
and in particular any later version will serve the purpose just as well.
The recommended way to obtain it is via [pyenv](https://github.com/pyenv/pyenv).

The experiments have been designed and tested under Linux.
However, we have some multi-platform build automation in place.
Everything should run on any recent Linux, MacOS X, and Windows setup.

### Reference machine

We provide a reference Travis CI configuration to maintain reproducibility over time.
While this image: [![Build Status](https://travis-ci.com/aPlacuzzi/Experiment-2020-FutureInternet-LoRa.svg?branch=master)](https://travis-ci.com/aPlacuzzi/Experiment-2020-FutureInternet-LoRa)
is green, the experiment is being maintained and,
by copying the configuration steps we perform for Travis CI in the `.travis.yml` file,
you should be able to re-run the experiment entirely.

### Automatic releases

Charts are remotely generated and made available on the project release page.
[The latest release](https://github.com/aPlacuzzi/Experiment-2020-FutureInternet-LoRa/releases/latest)
allows for quick retrieval of the latest version of the charts.

## Running the simulations

A graphical execution of the simulation can be started with the following command
`./gradlew runWithGUI`.
When the simulator is opened click the button <kbd>Open</kbd> and select the file `smartThermostats.xml`.
Then start the simulation by click the button <kbd>Timed run</kbd>.
With the sidebar to the right of the previous button is possible to speed up the simulation by reducing the number of frame drawed.
Windows users can use the `gradlew.bat` script as a replacement for `gradlew`.
A video of a graphical execution is available inside the directory `video` (the video is speeded up 16x).

The whole simulation batch can be executed by issuing `./gradlew batch`.
**Be aware that it may take a very long time**, from days to weeks, depending on your hardware.
With the actual configuration it executes 34560 simulations.
Each simulation is a different java process and the gradle task `batch` parallelize their execution using all the cores available. 

## Generating the charts

In order to speed up the process for those interested in observing and manipulating the existing data,
we provide simulation-generated data directly in the repository.
Generating the charts is matter of executing the `process.py` script.
The environment is designed to be used in conjunction with pyenv.

### Python environment configuration

The following guide will start from the assumption that pyenv is installed on your system.
First, install Python by issuing

``pyenv install --skip-existing 3.8.1``

Now, configure the project to be interpreted with exactly that version:

``pyenv local 3.8.1``

Update the `pip` package manager and install the required dependencies.

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

Finally, to allow the rendering of text with LaTeX are required the following dependency: LaTeX, dvipng, and Ghostscript (
see [matplotlib documentation](https://matplotlib.org/3.2.1/tutorials/text/usetex.html) for more info).
If you don't already have them, you can install them by issuing

```bash
apt-get update
apt-get install texlive-latex-base
apt-get install dvipng
apt-get install ghostscript
```


### Data processing and chart creation

This section assumes you correctly completed the required configuration described in the previous section.
In order for the script to execute, you only need to launch the actual process by issuing `python process.py`
