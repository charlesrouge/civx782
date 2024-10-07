# CIVX782
Tutorial resources for CIV4782-CIV6782 module "Resilient water infrastructure design"

## How to clone and update the repository to your Uni drive U:\Users\<username>\
where <username> is your uni login (e.g., ciXXXX)

=> In the Search bar of your machine, type `cmd` to open a command prompt.

=> Navigate to U:\Users\<username> using the `cd Users` then `cd <username>` prompts.

=> If it is the first time your are importing this folder type
`git clone https://github.com/charlesrouge/civx782.git`

=> If it is not navigate to the civx782 folder using `cd civx782`then
`git pull origin main`


## How to run tutorials on Jupyter Notebook

1) Open Jupyter Notebook and identify the directory it starts in. For most machines this is
C:\Users\<username>

2) Still from the command line, exit the civx782 folder to be in C:\Users\<username>, with 
`cd ..`

3) Copy the civx782 folder to that directory using:
`Xcopy civx782 C:\Users\<username> /E`
then press `D` if prompted to type `F` or `D`

4) The `civx782` folder should have magically appeared on the Jupyter Notebook main menu!


## How to install the virtual environment and run it on Jupyter Notebook

=> Open the directory containing the `environment.yml` file and type in terminal
`conda env create -f environment.yml`

=> Or if you cannot create the environment on the University machine, still create a new environment and work in it. This will keep track of libraries used and versions as you go!
`conda env create --name civx782`

=> type in terminal
`python -m ipykernel install --user --name=civx782`

=> in Jupyer Notebook click on `Python 3 (ipykernel)` in the top right, and replace with `civx782` as a kernel
