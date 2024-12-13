:grey_exclamation:**Installation**  
If you are a conda user you can create a new environment with all the dependencies using the environment.yml file using the command:
```
conda env create -f environment.yml
```
If you are not a conda user you can see all the necessary dependencies in the environment.yml file.

:grey_exclamation:**About this project**  
All the scripts are ready-to-use and work locally, saving the results locally in a structure described in the next section.
BE AWARE: all the simulations and data manipualtion generate GiB of files which you may not want to store on your local machine. If you intend to store everything in a dedicated
storage unit change the paths in the scripts you are using. This should be easy as all scripts have an "input_folder" and an "output_folder". We suggest to only change the root of the
input and output paths in a way the default data structure is preserved, otherwise you might have problems when running other scripts related.

:grey_exclamation:**Project local file organization**  
Main:
  - Scripts common through all methods
  - A folder to contain the graphs used for this work
  - One folder per method:
      - Scripts specific to the method
      - Folder for data storage:
          - one folder per each data type:
              - (if necesseray): one folder per each control parameter

:bangbang:**Important for collaborators**  
If you add packages to this project remember to update the environment.yml before pushing!
You can do that by exporting a new one with the command:
```
conda env export --from-history > environment.yml
```
This will overwrite the old file with a new one which is "personalized" to your pc. In order to make it usable by other you need to open the new environment.yml file
and manually remove the line "prefix: <your>/<specific>/<path>" .
In this way any other machine can use this file and store binaries in their default folders.
