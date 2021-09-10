# Data analysis
- Description: Smart Cities is a formulation given to cities that gather data in order to solve a problem. One of them is heat. Indeed construction materials dissipate hardly the heat making some areas very hot even at night. A solution given by researchers is trees and vegetation in a general sense to cool down specific areas. Our application is a way to give insight on where to put vegetation based on current temperature and vegetation index. The final product is a website, with position and date in input. A map is generated as output highlighting the areas where putting vegetation will have the most impact.
- Data Source: LANDSAT8 images database through Google Earth Engine API
- Type of analysis: Gathering and computing satellite images in order to show on a given city, which areas are the hottest relatively to their vegetation index.

# Startup the project

The initial setup.

Create virtualenv and install the project:
```bash
sudo apt-get install virtualenv python-pip python-dev
deactivate; virtualenv ~/venv ; source ~/venv/bin/activate ;\
    pip install pip -U; pip install -r requirements.txt
```

Unittest test:
```bash
make clean install test
```

Check for Smartrees in gitlab.com/{group}.
If your project is not set please add it:

- Create a new project on `gitlab.com/{group}/Smartrees`
- Then populate it:

```bash
##   e.g. if group is "{group}" and project_name is "Smartrees"
git remote add origin git@github.com:{group}/Smartrees.git
git push -u origin master
git push -u origin --tags
```

Functionnal test with a script:

```bash
cd
mkdir tmp
cd tmp
Smartrees-run
```

# Install

Go to `https://github.com/{group}/Smartrees` to see the project, manage issues,
setup you ssh public key, ...

Create a python3 virtualenv and activate it:

```bash
sudo apt-get install virtualenv python-pip python-dev
deactivate; virtualenv -ppython3 ~/venv ; source ~/venv/bin/activate
```

Clone the project and install it:

```bash
git clone git@github.com:{group}/Smartrees.git
cd Smartrees
pip install -r requirements.txt
make clean install test                # install and test
```
Functionnal test with a script:

```bash
cd
mkdir tmp
cd tmp
Smartrees-run
```
