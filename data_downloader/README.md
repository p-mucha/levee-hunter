# Geometry data downloader

## Introduction
`download_lidar_data.py` is a simple script to download geometry data from the national map using the [TNM access API](https://apps.nationalmap.gov/tnmaccess/#/product).

## How does the script work
The script will
1. Build a task list from the levee data:
    1. Load the levee data in `levees.gpkg`
    1. Obtain the `total_bounds` of all the levee data
    1. Divide `total_bounds` into smaller cells (while the size of the cells is defined by the `--cellSize` parameter)
    1. Check the overlap between cells and levees and only keep cells overlapping with at least one levee.
    1. The bounding box of each cell is used to define one task
1. Use the task list to query the download URLs:
    1. Each task forms a request to be sent to the TNM access API
    1. Each response from the may contain several items
    1. If the item has not been downloaded (by checking the `sourceId`), add the `downloadURL` of the item to the job list
    1. Several tasks are bundled together to form a chunk. The number of tasks is defined by '--chunkSize'.
1. Use the job list to download the data.
    1. The downloading starts after all the quries in each chunk are sent.
    1. The number of downloading jobs is controled by `--nJobs`.
    1. After all downloading jobs in one chunk are finished, the script starts to send queries for the next chunk.


## How to run the downloading script
### Preparations
- Modify `LEVEE_PATH` to point to the leeves.gpkg file.
- Modify the data format `PRODFORMAT` if needed.
- Modify `DATASETS` to select which dataset you would like to query.

### Run the script

Display the help messages:
```
python download_lidar_data.py -h
```

Do the full task generation and downloading as described above:
```
python download_lidar_data.py
```
The output path can be specificed with option `-o`.

If you only want to build a task list, but do not want to send queries or downlaod any data:
```
python download_lidar_data.py -g
```
This will generate a `task_list.txt` file in the output path.

You could modify the task list file and a modified list back to the script for downloading. The user task list takes priority and it will disable the internal task list generation. To feed a user task list file:
```
python download_lidar_data.py -l task_list.txt
```
If you want to send the queries and see the responses, but do not want to download any data:
```
python download_lidar_data.py --dryRun
```

If you do not want to use the total bounds of the levee data, you could provide a different total bounds `-b xmin ymin xmax ymax`, e.g.
```
python download_lidar_data.py -b -121.58826 37.96628 -120.51895 39.01853
```
The script will make cells from the user total bounds and ignore the total bounds of the levee data. After that it will perform the cell-levee overlap check as usual.


## Troubleshooting
**UserWarning:** Query returns 65 items, exceeding the limit of 50. Consider to descrease the cell size.

This happens when `cellSize` is too large. The response will only return the first 50 items if there are more than 50 in one query. Set a smaller `cellSize` to avoid this.

**Other exceptions**: The code will print out the request and the response. The response usually contain some error messages. You could also copy the request and and paste it in a web browser to get the response.