# FYDP

## Setup

To set up the project, run the following commands in your terminal:

* `pip install virtualenv`
* `pip install virtualenvwrapper`
* `export WORKON_HOME=~/Envs`
* `mkdir -p $WORKON_HOME`

As part of the install instructions for virtualenvwrapper, you need to add this to your ~/.bash_profile file:

* `source /usr/local/bin/virtualenvwrapper.sh`

Then, run this command in your terminal:

* `source ~/.bash_profile`

Virtualenvwrapper makes it easy to keep several virtual environments. To make things even easier, it's suggested that you add these lines to your `~/.bash_profile` file (http://blog.doughellmann.com/2010/01/virtualenvwrapper-tips-and-tricks.html).

```
alias v='workon'
alias v.deactivate='deactivate'
alias v.mk='mkvirtualenv --no-site-packages'
alias v.mk_withsitepackages='mkvirtualenv'
alias v.rm='rmvirtualenv'
alias v.switch='workon'
alias v.add2virtualenv='add2virtualenv'
alias v.cdsitepackages='cdsitepackages'
alias v.cd='cdvirtualenv'
alias v.lssitepackages='lssitepackages'
```

Then, create a virtual environment by running this in your terminal:

* `v.mk venv`

or 

* `mkvirtualenv --no-site-packages venv`

if you haven't updated the aliases in your `~/.bash_profile` file.

This creates and virtual environment and makes it active. To deactivate it, you can simply type:

* `deactivate`

Next, install the requirements for this project by running the following command in the FYDP project directory:

* `pip install -r requirements.txt`

You're done with the setup!

## Running the project

To work with the project, first activate the virtual environment:

* `v venv`

Then, run main.py:

* `src/bin/run`

To deactivate it, you can simply type:

* `deactivate`

## Troubleshooting

If you run into errors with PIL while installing the requirements, run:

* `pip install PIL --allow-external PIL --allow-unverified PIL`

and run `pip install -r requirements.txt` again.
