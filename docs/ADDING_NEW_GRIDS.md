
# Adding new grids in an AEZ for improving the LULC pipeline

The process of adding new grids in the AEZ is to help the LULC pipeline download data from these grids, process it, and then use the samples generated to train a new classifier for that specific AEZ.

## Step 1: Select grids from the Tile Explorer app

First step is to load the link:  
https://raman-461708.projects.earthengine.app/view/tile-explorer

This has the app where you can select the grids and it will give the `grid_ids` along with the AEZ no.

To use it, first open the link. This will show the app interface.

- Cyan colored grids are the grids which are unprocessed and so available for selection.
- Yellow colored grids are those which are already processed so they can't be selected.
- You can switch on the LULC from the left pane if you want to see the LULC v4 underneath the grids.

To select grids, you need to draw a rectangle on the area whose LULC is not accurate. When you draw the rectangle, it will select all the grids which intersect with the rectangle and show the grids in red.

On the left pane it will show the `grid_ids` along with the AEZ no.

You can draw multiple rectangles at once and it will select multiple grids and all the `grid_ids` will be shown on the left.

The preprocessed yellow grids will automatically be unselected.

To reset the selection there is a reset button given.

## Step 2: Save selected grid IDs in a CSV file

Once you have the `grid_ids`, you can copy them in a file and name it for example `grid_ids.csv`.

Then paste it inside a folder named `AEZ_<no>`.

The folder should also have `point.csv` which is created when you first execute the AEZ level data generation script `1_local_compute_all.py`.

## Step 3: Run the AEZ level processing script for selected grids

Once you add the csv, you can run this script with changed parameters to only download and process the `grid_ids` selected.

Use this command to run on the `grid_ids`:

```bash
python3 "src (AEZ level)/1_local_compute_all.py" grid --aez 6 --grid-file "grid_id.csv" --base-dir "data/data_AEZ"
````

Here we are doing it for AEZ no 6.

* `--aez 6` means AEZ number 6
* `--grid-file "grid_id.csv"` is the csv file containing the selected grid ids
* `--base-dir "data/data_AEZ"` is the directory in which you want to download and keep all the data

Our code will first create a `status.csv` file where these grids are further broken down to 4 tiles each and written to a status file which also has other variables like download, segment, etc. that save the current state of execution for how much of the execution has taken place.

## Step 4: Only generate the status.csv without running the full process

To only generate and see the `status.csv` before running all the process, you can execute following command:

```bash
python3 "src (AEZ level)/1_local_compute_all.py" grid --aez 6 --grid-file "6_grids.csv" --base-dir "data/data_AEZ" --skip-run
```

## Step 5: Generate samples from the boundaries

The next step is to execute second script by the name `2_sampling_locally_workers.py` which will create samples form the boundaries.

To execute it use the following command:

```bash
python3 "src (AEZ level)/2_sampling_locally_workers.py" --aez-root "data/data_AEZ_increment" --boundary-root "data/data_AEZ_increment" single --aez 1 --collate-out "gee_samples.csv"
```

It will create `gee_samples.csv` inside the `AEZ_<no>` folder.

These samples can be used in the LULC creation to train better models with samples from the region where inaccuracy was reported.

## Step 6: Create the LULC

The last step is to to create the LULC, which I will explain further.


